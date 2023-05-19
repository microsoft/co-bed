import argparse
import math
import os
import pickle
import time

import mlflow
import pandas as pd
import pyro
import torch
from joblib import Parallel, delayed
from tqdm import trange

from bayesian_simulators.continuous_scalar_context_scalar_treatment import (
    ContinuousContextAndTreatment,
)
from bayesian_simulators.treatment_policy import IdentityTreatmentPolicy
from nn_modules.critics import DotProductCritic
from nn_modules.critic_encoder import Encoder
from objectives.implicit_mi import InfoNCE


def train_net(
    context_obs,
    context_test,
    hidden_dim,
    encoding_dim,
    batch_size,
    seed,
    learning_rate,
    gamma,
    num_steps,
    annealing_frequency,
    logging_frequency,
    iteration,
    optimise_design,
    ucb_baseline,
    thompson_sampling_baseline,
    device,
    return_max_reward=True,
    sd_init_design=0.2,
    run_id="",
    experiment_id="",
):
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)

        design_dim = context_obs.shape[-1]
        input_dim_enc_x = design_dim
        input_dim_enc_y = context_test.shape[-1]

        if ucb_baseline is not None:
            init_design = compute_ucb_design(
                context=context_obs,
                alpha=ucb_baseline,
                batch_size_treatment=500,
                batch_size_parameters=500,
                device=device,
            )
            optimise_design = False

        elif thompson_sampling_baseline:
            init_design = compute_thompson_sampling_design(
                context=context_obs, grid_size_treatment=500, device=device
            )
            optimise_design = False

        else:
            init_design = (
                torch.distributions.Normal(0.0, sd_init_design)
                .sample([design_dim])  # type: ignore
                .to(device)
            )

        enc_x = Encoder(
            input_dim=input_dim_enc_x, hidden_dim=hidden_dim, encoding_dim=encoding_dim
        ).to(device)

        enc_y = Encoder(
            input_dim=input_dim_enc_y, hidden_dim=hidden_dim, encoding_dim=encoding_dim
        ).to(device)
        critic = DotProductCritic(enc_x, enc_y, scale=False).to(device)

        PRIOR = {
            f"psi{i}": pyro.distributions.Uniform(  # type: ignore
                torch.tensor([0.1]).to(device), torch.tensor([1.1]).to(device)
            ).to_event(1)
            for i in range(4)
        }

        cct = ContinuousContextAndTreatment(
            treatment_policy=IdentityTreatmentPolicy(
                init_design, learnable=optimise_design
            ),
            priors_on_parameters=PRIOR,
            cost_weight=0.1,
            return_max_reward=return_max_reward,
        ).to(device)
        mi_instance = InfoNCE(cct, critic, batch_size)

        base_optim = torch.optim.Adam

        def separate_learning_rate(module_name, param_name):
            if module_name == "critic_net":
                return {"lr": learning_rate}
            elif param_name == "treatment":
                return {"lr": learning_rate * 1}
            else:
                raise NotImplementedError()

        optim = pyro.optim.ExponentialLR(  # type: ignore
            {
                "optimizer": base_optim,
                "optim_args": separate_learning_rate,
                "gamma": gamma,
            }
        )

        num_steps_range = trange(
            1, num_steps + 1, desc="Loss: 0.000 ", position=0, leave=True
        )
        scheduler_step = getattr(optim, "step", False)
        designs = []

        for i in num_steps_range:
            loss = mi_instance.train_step(
                optim, context_obs=context_obs, context_test=context_test
            )
            if scheduler_step and i % annealing_frequency == 0:
                optim.step()
            if (i - 1) % logging_frequency == 0:
                # check if loss (a number) is nan:
                num_steps_range.set_description(f"Loss: {loss:.3f} ")
                mlflow.log_metric(
                    f"loss_{iteration}",
                    mi_instance.loss(
                        context_obs=context_obs, context_test=context_test
                    ),
                    step=i,
                )
                design = cct.treatment_policy.treatment.detach().cpu()  # type: ignore
                designs.append([i - 1] + [*design.numpy()])

        mi_mean, mi_se = mi_instance.evaluate_mi(
            n_reps=100, context_obs=context_obs, context_test=context_test
        )
        mlflow.log_metric(f"mi_mean_{iteration}", mi_mean)
        design_df = pd.DataFrame(
            designs, columns=["step"] + [f"design{d}" for d in range(1, design_dim + 1)]
        ).set_index("step")
        design_df.columns.name = "design"

        return {
            "context_obs": context_obs,
            "context_test": context_test,
            "design": mi_instance.optimal_design,
            "mi_mean": mi_mean,
            "mi_se": mi_se,
        }


def compute_ucb_design(
    context,
    alpha,
    batch_size_treatment=300,
    batch_size_parameters=500,
    device="cuda",
    return_max_reward=True,
):
    pyro.clear_param_store()

    batch_size_context = context.shape[0]
    # expand
    context_expanded = (
        context.unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size_parameters, batch_size_context, batch_size_treatment)
        .to(device)
    )
    treatment = torch.linspace(-4, 4, batch_size_treatment).to(device)
    treatment_expanded = treatment.expand(
        batch_size_parameters, batch_size_context, batch_size_treatment
    )
    PRIOR = {
        f"psi{i}": pyro.distributions.Uniform(  # type: ignore
            torch.tensor([0.1]).to(device), torch.tensor([1.1]).to(device)
        ).to_event(1)
        for i in range(4)
    }
    cct = ContinuousContextAndTreatment(
        priors_on_parameters=PRIOR,
        treatment_policy=IdentityTreatmentPolicy(treatment_expanded, learnable=False),
        cost_weight=0.1,
        return_max_reward=return_max_reward,
    ).to(device)

    parameters_dict = cct._vectorize(
        cct.sample_parameters, [batch_size_parameters, 1], "expand_psi"
    )()
    y_means = cct.sample_observational_data(
        parameters_dict, context_expanded, return_mean=True
    )
    assert y_means.shape == torch.Size(
        [batch_size_parameters, batch_size_context, batch_size_treatment]
    )
    mean = y_means.mean(0)
    sd = y_means.std(0)
    ucb = mean + alpha * sd
    _, idx = ucb.max(-1)
    optimal_design = treatment[idx]
    assert optimal_design.shape == context.shape
    return optimal_design


def compute_thompson_sampling_design(
    context, grid_size_treatment=300, device="cuda", return_max_reward=True
):
    pyro.clear_param_store()
    batch_size_context = context.shape[0]
    # expand
    context_expanded = (
        context.unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size_context, batch_size_context, grid_size_treatment)
        .to(device)
    )
    treatment = torch.linspace(-4, 4, grid_size_treatment).to(device)
    treatment_expanded = treatment.expand(
        batch_size_context, batch_size_context, grid_size_treatment
    )
    PRIOR = {
        f"psi{i}": pyro.distributions.Uniform(  # type: ignore
            torch.tensor([0.1]).to(device), torch.tensor([1.1]).to(device)
        ).to_event(1)
        for i in range(4)
    }
    cct = ContinuousContextAndTreatment(
        priors_on_parameters=PRIOR,
        treatment_policy=IdentityTreatmentPolicy(treatment_expanded, learnable=False),
        cost_weight=0.1,
        return_max_reward=return_max_reward,
    ).to(device)

    parameters_dict = cct._vectorize(
        cct.sample_parameters, [batch_size_context, 1], "expand_psi"
    )()
    y_means = cct.sample_observational_data(
        parameters_dict, context_expanded, return_mean=True
    )
    assert y_means.shape == torch.Size(
        [batch_size_context, batch_size_context, grid_size_treatment]
    )
    _, idx = y_means.max(-1)
    idx = torch.diag(idx)
    optimal_design = treatment[idx]
    assert optimal_design.shape == context.shape
    return optimal_design


def calculate_regret(
    context_obs,
    design,
    context_test,
    num_true_models_to_sample,
    seed=None,
    return_max_reward=True,
    inference="snis",
):
    # Variables in ALL_CAPS indicate interaction with the Real World Environment simulator.
    if seed is not None:
        pyro.set_rng_seed(seed)

    # post_samples_list = []
    regret_list = []
    mean_regret_list = []
    TRUE_PARAMS_list = []
    TRUE_OPTIMAL_TREATMENT_list = []
    TRUE_MAX_REWARD_list = []
    learnt_max_reward_list = []
    learnt_optimal_treatment_list = []
    posterior_mean_list = []
    posterior_sd_list = []

    mean_regret = torch.zeros(num_true_models_to_sample)
    psi_recovery = torch.zeros(num_true_models_to_sample)
    reward_recovery = torch.zeros(num_true_models_to_sample)
    treatment_recovery = torch.zeros(num_true_models_to_sample)
    psi_in_2sd = torch.zeros(num_true_models_to_sample)

    models_range = trange(
        0, num_true_models_to_sample, desc="Regret: 0.000 ", position=0, leave=True
    )

    for i in models_range:
        pyro.clear_param_store()
        PRIOR = {
            f"psi{i}": pyro.distributions.Uniform(  # type: ignore
                torch.tensor([0.1]), torch.tensor([1.1])
            ).to_event(1)
            for i in range(4)
        }
        CCT_TRUE = ContinuousContextAndTreatment(
            treatment_policy=IdentityTreatmentPolicy(design, learnable=False),
            priors_on_parameters=PRIOR,
            return_max_reward=return_max_reward,
        )
        TRUE_MODEL_PARAMETERS = CCT_TRUE.sample_parameters()
        (
            TRUE_MAX_REWARD,
            TRUE_OPTIMAL_TREATMENT,
        ) = CCT_TRUE.calculate_conditional_max_reward(
            TRUE_MODEL_PARAMETERS, context_test
        )
        OBSERVATIONS = CCT_TRUE.sample_observational_data(
            TRUE_MODEL_PARAMETERS, context=context_obs
        )
        data_for_inference = {"y": OBSERVATIONS, "args": [context_obs]}
        ## store the true data
        TRUE_PARAMS_list.append(TRUE_MODEL_PARAMETERS)
        TRUE_OPTIMAL_TREATMENT_list.append(TRUE_OPTIMAL_TREATMENT)
        TRUE_MAX_REWARD_list.append(TRUE_MAX_REWARD)

        # Run inference to obtain posterior model
        # Inference is ran without interacting with the TRUE MODEL
        cct_model = ContinuousContextAndTreatment(
            treatment_policy=IdentityTreatmentPolicy(design, learnable=False),
            priors_on_parameters=PRIOR,
            return_max_reward=return_max_reward,
        )
        if inference == "hmc":
            cct_model.infer_model(
                inference,
                observed_data=data_for_inference,
                num_samples=2000,
                num_chains=12,
            )
        else:
            cct_model.infer_model(
                inference, observed_data=data_for_inference, num_samples=10000
            )

        post_samples = cct_model._vectorize(
            cct_model.sample_parameters, [1000], "posterior"
        )()
        posterior_mean = {key: value.mean() for key, value in post_samples.items()}

        posterior_sd = {key: value.std() for key, value in post_samples.items()}
        truth_within_2sd_proportion = [
            1.0
            if TRUE_MODEL_PARAMETERS[key] > posterior_mean[key] - 2 * posterior_sd[key]
            and TRUE_MODEL_PARAMETERS[key] < posterior_mean[key] + 2 * posterior_sd[key]
            else 0.0
            for key in TRUE_MODEL_PARAMETERS.keys()
        ]
        # find the optimal treatment for the posterior mean and the test context
        max_rewards, optimal_treatments = cct_model.calculate_conditional_max_reward(
            post_samples, context_test
        )
        marginal_max_reward = max_rewards.mean(0)
        marginal_optimal_treatment = optimal_treatments.mean(0)

        psi_recovery[i] = (
            torch.stack(
                [
                    posterior_mean[key] - TRUE_MODEL_PARAMETERS[key]
                    for key in posterior_mean.keys()
                ]
            )
            .pow(2)
            .mean()
            .item()
        )
        reward_recovery[i] = (
            (marginal_max_reward - TRUE_MAX_REWARD).pow(2).mean().item()
        )
        treatment_recovery[i] = (
            (TRUE_OPTIMAL_TREATMENT - marginal_optimal_treatment).pow(2).mean().item()
        )
        psi_in_2sd[i] = sum(truth_within_2sd_proportion) / len(
            truth_within_2sd_proportion
        )

        posterior_mean_list.append(posterior_mean)
        posterior_sd_list.append(posterior_sd)
        learnt_max_reward_list.append(marginal_max_reward)
        learnt_optimal_treatment_list.append(marginal_optimal_treatment)

        # Use the TRUE MODEL to obtain the "true" (i.e. average realised) reward for the learnt best action
        # this involves interaction with the TRUE MODEL
        CCT_TRUE.treatment_policy = IdentityTreatmentPolicy(
            marginal_optimal_treatment, learnable=False
        )  # ! update the treatment
        reward_obtained_under_true_model = CCT_TRUE.sample_observational_data(
            TRUE_MODEL_PARAMETERS, context_test, return_mean=True
        )
        regret = TRUE_MAX_REWARD - reward_obtained_under_true_model
        regret_list.append(regret)
        mean_regret[i] = regret.mean().item()
        models_range.set_description("Regret: {:.3f} ".format(mean_regret[i]))

    return {
        "mean_regret_list": mean_regret_list,
        "regret_list": regret_list,
        "TRUE_PARAMS_list": TRUE_PARAMS_list,
        "TRUE_OPTIMAL_TREATMENT_list": TRUE_OPTIMAL_TREATMENT_list,
        "TRUE_MAX_REWARD_list": TRUE_MAX_REWARD_list,
        "learnt_max_reward_list": learnt_max_reward_list,
        "learnt_optimal_treatment_list": learnt_optimal_treatment_list,
        "posterior_mean_list": posterior_mean_list,
        "posterior_sd_list": posterior_sd_list,
        "mean_regret": mean_regret,
        "psi_recovery": psi_recovery,
        "reward_recovery": reward_recovery,
        "treatment_recovery": treatment_recovery,
        "psi_in_2sd": psi_in_2sd,
        "marginal_optimal_treatment": marginal_optimal_treatment,  # type: ignore
    }


def main(
    num_steps,
    seed,
    batch_size,
    hidden_dim,
    encoding_dim,
    learning_rate,
    gamma,
    device,
    design_dim,
    num_jobs,
    optimise_design,
    ucb_baseline=None,
    thompson_sampling_baseline=False,
    num_true_models_to_sample=20,
    return_max_reward=True,
    sd_init_design=0.2,
    inference="snis",
):
    mlflow.set_experiment("Illustrative: Continuous Actions")
    mlflow.start_run()

    # These could be added as args
    annealing_frequency = 1000
    logging_frequency = 10
    # adjust hidden dim

    hidden_dim = [design_dim * 2, hidden_dim, hidden_dim // 2]

    if isinstance(seed, int):
        seed = [seed]

    os.makedirs(f"outputs/logging", exist_ok=True)
    mlflow.log_params(
        {
            "hidden_dim_modified": hidden_dim,
            "encoding_dim": encoding_dim,
            "seed": seed,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "optimise_design": optimise_design,
        }
    )

    context_obs = torch.linspace(-4.0, 4.0, steps=design_dim).to(device)
    test_offset = (context_obs[1] - context_obs[0]) / 2
    context_test = context_obs[:-1] + test_offset

    results = Parallel(n_jobs=num_jobs, backend="loky")(
        delayed(train_net)(
            context_obs=context_obs,
            context_test=context_test,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            batch_size=batch_size,
            seed=seed,
            learning_rate=learning_rate,
            gamma=gamma,
            num_steps=num_steps,
            annealing_frequency=annealing_frequency,
            logging_frequency=logging_frequency,
            iteration=j,
            optimise_design=optimise_design,
            ucb_baseline=ucb_baseline,
            thompson_sampling_baseline=thompson_sampling_baseline,
            device=device,
            return_max_reward=return_max_reward,
            sd_init_design=sd_init_design,
            run_id=mlflow.active_run().info.run_id,  # type: ignore
            experiment_id=mlflow.get_experiment_by_name(
                "Illustrative: Continuous Actions"
            ).experiment_id,  # type: ignore
        )
        for j, seed in enumerate(seed)
    )

    assert results is not None, "Error. Results is None."

    context_obs_df = pd.DataFrame(
        [res.get("context_obs").cpu().numpy() for res in results],
        index=seed,
        columns=[f"context_obs{i}" for i in range(1, design_dim + 1)],
    )
    context_test_df = pd.DataFrame(
        [res.get("context_test").cpu().numpy() for res in results],
        index=seed,
        columns=[f"context_test{i}" for i in range(1, design_dim)],
    )
    design_df = pd.DataFrame(
        [res.get("design").cpu().numpy() for res in results],
        index=seed,
        columns=[f"design{i}" for i in range(1, design_dim + 1)],
    )
    mi_mean = pd.DataFrame(
        [res.get("mi_mean") for res in results], index=seed, columns=["mi_mean"]
    )
    mi_se = pd.DataFrame(
        [res.get("mi_se") for res in results], index=seed, columns=["mi_sd"]
    )
    df = pd.concat([context_obs_df, context_test_df, design_df, mi_mean, mi_se], axis=1)
    df.index.name = "seed"
    df["design_dim"] = design_dim

    df.to_csv(f"outputs/logging/results_table.csv")
    mlflow.log_metric("mi_across_seeds", df["mi_mean"].mean(skipna=True))

    # REGRETS
    # make sure everything is on CPU!!
    eval_results = Parallel(n_jobs=num_jobs, backend="loky")(
        delayed(calculate_regret)(
            context_obs=res.get("context_obs").cpu(),
            design=res.get("design").cpu(),
            context_test=res.get("context_test").cpu(),
            seed=seed[i],
            return_max_reward=return_max_reward,
            num_true_models_to_sample=num_true_models_to_sample,
            inference=inference,
        )
        for i, res in enumerate(results)
    )
    with open(f"outputs/logging/eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)

    def calculate_mean_se(results_dict, names):
        means = {f"{name}_mean": results_dict[name].mean().item() for name in names}
        ses = {
            f"{name}_se": results_dict[name].std().item()
            / math.sqrt(len(results_dict[name]))
            for name in names
        }
        return {**means, **ses}

    metrics = pd.DataFrame(
        [
            calculate_mean_se(
                eval_res,
                [
                    "mean_regret",
                    "psi_recovery",
                    "reward_recovery",
                    "treatment_recovery",
                    "psi_in_2sd",
                ],
            )
            for eval_res in eval_results  # type: ignore
        ]
    )
    metrics.index = df.index
    df = pd.concat([df, metrics], axis=1)

    mlflow.log_metric("regret", df["mean_regret_mean"].mean(skipna=True))
    mlflow.log_metric("psi_recovery", df["psi_recovery_mean"].mean(skipna=True))
    mlflow.log_metric("reward_recovery", df["reward_recovery_mean"].mean(skipna=True))
    mlflow.log_metric(
        "treatment_recovery", df["treatment_recovery_mean"].mean(skipna=True)
    )

    df.to_csv(f"outputs/logging/results_table.csv")
    mlflow.log_artifact(f"outputs/logging/results_table.csv")

    mlflow.end_run()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous actions")
    parser.add_argument("-o", type=str)

    parser.add_argument("--seed-reps", default=3, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--num-jobs", default=-1, type=int)
    parser.add_argument("--num-steps", default=500, type=int)
    parser.add_argument("--design-dim", default=5, type=int)
    parser.add_argument("--optimise-design", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--ucb-baseline", default=None, type=float)
    parser.add_argument("--thompson-sampling-baseline", action="store_true")
    parser.add_argument("--sd-init-design", default=0.2, type=float)
    parser.add_argument("--num-true-models-to-sample", default=3, type=int)

    args = parser.parse_args()
    seed = [int(torch.rand(tuple()) * 2**30) for _ in range(args.seed_reps)]

    main(
        num_steps=args.num_steps,
        seed=seed,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        device=args.device,
        design_dim=args.design_dim,
        num_jobs=args.num_jobs,
        optimise_design=args.optimise_design,
        ucb_baseline=args.ucb_baseline,
        thompson_sampling_baseline=args.thompson_sampling_baseline,
        return_max_reward=True,
        sd_init_design=args.sd_init_design,
        num_true_models_to_sample=args.num_true_models_to_sample,
        inference="snis",
    )
