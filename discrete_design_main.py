import argparse
import math
import os
import pickle

import mlflow
import pandas as pd
import pyro
import pyro.distributions as dist

import torch
from joblib import Parallel, delayed
from tqdm import trange

from bayesian_simulators.continuous_scalar_context_discrete_treatment import (
    ContinuousContextDiscreteTreatment,
)
from bayesian_simulators.treatment_policy import (
    GumbelSoftMaxPolicy,
    IdentityTreatmentPolicy,
)
from nn_modules.critics import DotProductCritic
from nn_modules.critic_encoder import Encoder
from objectives.implicit_mi import InfoNCE


def get_prior(device):
    priors_on_parameters = {
        "c_neg": dist.Normal(
            torch.tensor([[5.0, 5.0, -2.0, -7.0]]).to(device),
            torch.tensor([[1.5, 3.0, 1.1, 1.1]]).to(device),
        ).to_event(2),
        "c_pos": dist.Normal(
            torch.tensor([[15.0, 15.0, -1.0, 3.0]]).to(device),
            torch.tensor([[1.5, 3.0, 1.1, 1.1]]).to(device),
        ).to_event(2),
    }
    return priors_on_parameters


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
    tau,
    hard,
    temp_annealing_frequency=5000,
    temp_gamma=0.5,
    run_id="",
    experiment_id="",
):
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        SWITCH_TO_HARD = int(num_steps * 0.8)

        batch_norm = True
        design_dim = context_obs.shape[0]
        input_dim_enc_x = design_dim  #!!
        input_dim_enc_y = context_test.shape[0]

        enc_x = Encoder(
            input_dim=input_dim_enc_x,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            batch_norm=batch_norm,
        ).to(device)

        enc_y = Encoder(
            input_dim=input_dim_enc_y,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            batch_norm=batch_norm,
        ).to(device)
        critic = DotProductCritic(enc_x, enc_y, scale=False).to(device)

        priors_on_parameters = get_prior(device=device)
        treatment_dim = priors_on_parameters["c_pos"].sample().shape[-1]

        if optimise_design:
            log_probs = 0.01 * torch.ones((design_dim, treatment_dim)).to(device)
            policy = GumbelSoftMaxPolicy(
                log_prob=log_probs, tau=tau, hard=hard, learnable=True
            ).to(device)
        else:
            if ucb_baseline is not None:
                init_design = compute_ucb_design(
                    context=context_obs.squeeze(-1),
                    alpha=ucb_baseline,
                    batch_size_parameters=500,
                    device=device,
                )
                optimise_design = False

            elif thompson_sampling_baseline:
                init_design = compute_thompson_sampling_design(
                    context=context_obs, device=device
                )
                optimise_design = False

            else:
                # random binary policy: sample once
                init_design = torch.distributions.Multinomial(
                    probs=torch.ones((design_dim, treatment_dim)).to(device)
                ).sample()
            policy = IdentityTreatmentPolicy(treatment=init_design, learnable=False).to(
                device
            )

        model = ContinuousContextDiscreteTreatment(
            treatment_policy=policy, priors_on_parameters=priors_on_parameters
        ).to(device)
        mi_instance = InfoNCE(model, critic, batch_size)

        def separate_learning_rate(module_name, param_name):
            if module_name == "critic_net":
                return {"lr": learning_rate}
            elif module_name == "treatment_policy":
                return {"lr": learning_rate * 10}
            else:
                raise NotImplementedError()

        base_optim = torch.optim.Adam
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
        logprobs = []
        designs = []

        model.treatment_policy = policy

        for i in num_steps_range:
            loss = mi_instance.train_step(
                optim, context_obs=context_obs, context_test=context_test
            )
            if scheduler_step and i % annealing_frequency == 0:
                optim.step()

            if (
                hasattr(model.treatment_policy, "tau")
                and i % temp_annealing_frequency == 0
            ):
                model.treatment_policy.tau = (
                    model.treatment_policy.tau * temp_gamma
                )  # type: ignore

            if hasattr(model.treatment_policy, "hard") and i == SWITCH_TO_HARD:
                model.treatment_policy.hard = True  # type: ignore

            if (i - 1) % logging_frequency == 0:
                # check if loss (a number) is nan:
                num_steps_range.set_description("Loss: {:.3f} ".format(loss))
                mlflow.log_metric(
                    f"loss_{iteration}",
                    mi_instance.loss(
                        context_obs=context_obs, context_test=context_test
                    ),
                    step=iteration,
                )
                design_onehot = model.treatment_policy.treatment.detach()  # type: ignore
                design = design_onehot.max(-1).indices.cpu().numpy()
                designs.append([i - 1] + [*design])
                if hasattr(model.treatment_policy, "log_prob"):
                    logprobs.append(
                        model.treatment_policy.log_prob.detach().cpu().clone(),  # type: ignore
                    )

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


def compute_ucb_design(context, alpha, batch_size_parameters=500, device=None):
    pyro.clear_param_store()
    prior = get_prior(device=device)
    treatment_dim = prior["c_pos"].sample().shape[-1]
    batch_size_context = context.shape[0]
    # expand
    context_expanded = (
        context.expand(batch_size_parameters, batch_size_context)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    treatment = torch.eye(treatment_dim).to(device)
    treatment_expanded = treatment.expand(
        batch_size_parameters, batch_size_context, treatment_dim, treatment_dim
    )
    model = ContinuousContextDiscreteTreatment(
        priors_on_parameters=prior,
        treatment_policy=IdentityTreatmentPolicy(treatment_expanded, learnable=False),
    ).to(device)
    parameters_dict = model._vectorize(
        model.sample_parameters, [batch_size_parameters, 1], "sample_psi"
    )()
    with torch.no_grad():
        y_means = model.sample_observational_data(
            parameters_dict, context_expanded, return_mean=True
        )

    assert y_means.shape == torch.Size(
        [batch_size_parameters, batch_size_context, treatment_dim]
    )
    mean = y_means.mean(0)
    sd = y_means.std(0)
    ucb = mean + alpha * sd
    _, idx = ucb.max(-1)
    optimal_design = treatment[idx, :]
    assert optimal_design.shape == torch.Size([context.shape[0], treatment_dim])
    return optimal_design


def compute_thompson_sampling_design(context, device=None):
    pyro.clear_param_store()
    prior = get_prior(device=device)
    treatment_dim = prior["c_pos"].sample().shape[-1]
    batch_size_context = context.shape[0]
    # expand
    context_expanded = (
        context.expand(batch_size_context, batch_size_context)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    treatment = torch.eye(treatment_dim).to(device)
    treatment_expanded = treatment.expand(
        batch_size_context, batch_size_context, treatment_dim, treatment_dim
    )
    model = ContinuousContextDiscreteTreatment(
        priors_on_parameters=prior,
        treatment_policy=IdentityTreatmentPolicy(treatment_expanded, learnable=False),
    ).to(device)
    parameters_dict = model._vectorize(
        model.sample_parameters, [batch_size_context, 1], "sample_psi"
    )()
    with torch.no_grad():
        y_means = model.sample_observational_data(
            parameters_dict, context_expanded, return_mean=True
        )

    assert y_means.shape == torch.Size(
        [batch_size_context, batch_size_context, treatment_dim]
    )
    _, idx = y_means.max(-1)
    idx = torch.diag(idx)
    optimal_design = treatment[idx, :]
    assert optimal_design.shape == torch.Size([context.shape[0], treatment_dim])
    return optimal_design


def calculate_regret(
    context_obs,
    design,
    context_test,
    num_true_models_to_sample,
    seed=None,
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

    mean_regret = []  # torch.zeros(num_true_models_to_sample)
    psi_recovery = []  # torch.zeros(num_true_models_to_sample)
    reward_recovery = []  # torch.zeros(num_true_models_to_sample)
    treatment_recovery = []  # torch.zeros(num_true_models_to_sample)

    models_range = trange(
        0, num_true_models_to_sample, desc="Regret: -1.000 ", position=0, leave=True
    )

    if num_true_models_to_sample == 0:
        return

    for _ in models_range:
        pyro.clear_param_store()
        prior = get_prior(device="cpu")
        MODEL_TRUE = ContinuousContextDiscreteTreatment(
            treatment_policy=IdentityTreatmentPolicy(design, learnable=False),
            priors_on_parameters=prior,
        )
        TRUE_MODEL_PARAMETERS = MODEL_TRUE.sample_parameters()
        (
            TRUE_MAX_REWARD,
            TRUE_OPTIMAL_TREATMENT,
        ) = MODEL_TRUE.calculate_conditional_max_reward(
            TRUE_MODEL_PARAMETERS, context_test
        )
        OBSERVATIONS = MODEL_TRUE.sample_observational_data(
            TRUE_MODEL_PARAMETERS, context=context_obs
        )
        data_for_inference = {"y": OBSERVATIONS, "args": [context_obs]}
        ## store the true data
        TRUE_PARAMS_list.append(TRUE_MODEL_PARAMETERS)
        TRUE_OPTIMAL_TREATMENT_list.append(TRUE_OPTIMAL_TREATMENT)
        TRUE_MAX_REWARD_list.append(TRUE_MAX_REWARD)

        # Run inference to obtain posterior model
        # Inference is ran without interacting with the TRUE MODEL
        model = ContinuousContextDiscreteTreatment(
            treatment_policy=IdentityTreatmentPolicy(design, learnable=False),
            priors_on_parameters=prior,
        )
        model.infer_model(inference, observed_data=data_for_inference, num_samples=5000)

        try:
            post_samples = model._vectorize(
                model.sample_parameters, [2000], "posterior"
            )()
        except:
            continue

        posterior_mean = {key: value.mean(0) for key, value in post_samples.items()}

        posterior_sd = {key: value.std(0) for key, value in post_samples.items()}
        # find the optimal treatment for the posterior mean and the test context
        max_rewards, _ = model.calculate_conditional_max_reward(
            post_samples, context_test
        )
        marginal_max_reward = max_rewards.mean(0)
        # USE UCB 0.0 design
        marginal_optimal_treatment = compute_ucb_design(
            context_test.squeeze(-1), 0.0, device="cpu"
        )

        psi_recovery.append(
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
        reward_recovery.append(
            (marginal_max_reward - TRUE_MAX_REWARD).pow(2).mean().item()
        )
        treatment_recovery.append(
            (
                (
                    TRUE_OPTIMAL_TREATMENT != marginal_optimal_treatment.max(1).indices
                ).sum()
                / TRUE_OPTIMAL_TREATMENT.shape[0]
            ).item()
        )

        posterior_mean_list.append(posterior_mean)
        posterior_sd_list.append(posterior_sd)
        # post_samples_list.append(post_samples)
        learnt_max_reward_list.append(marginal_max_reward)
        learnt_optimal_treatment_list.append(marginal_optimal_treatment)

        # Use the TRUE MODEL to obtain the "true" (i.e. average realised) reward for the learnt best action
        # this involves interaction with the TRUE MODEL
        # ! update the treatment
        MODEL_TRUE.treatment_policy = IdentityTreatmentPolicy(
            marginal_optimal_treatment, learnable=False
        )
        reward_obtained_under_true_model = MODEL_TRUE.sample_observational_data(
            TRUE_MODEL_PARAMETERS, context_test, return_mean=True
        )
        regret = TRUE_MAX_REWARD - reward_obtained_under_true_model
        regret_list.append(regret)
        mean_regret.append(regret.mean().item())
        models_range.set_description("Regret: {:.3f} ".format(regret.mean().item()))

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
        "mean_regret": torch.tensor(mean_regret),
        "psi_recovery": torch.tensor(psi_recovery),
        "reward_recovery": torch.tensor(reward_recovery),
        "treatment_recovery": torch.tensor(treatment_recovery),
    }


def main(
    seed,
    device,
    design_dim,
    num_jobs,
    hidden_dim,
    encoding_dim,
    batch_size,
    learning_rate,
    gamma,
    optimise_design,
    num_steps,
    ucb_baseline,
    thompson_sampling_baseline,
    num_true_models_to_sample,
    tau=0.1,
    hard=False,
):
    mlflow.set_experiment("Illustrative: Discrete Actions")
    mlflow.start_run()
    pyro.clear_param_store()
    os.makedirs("outputs/logging", exist_ok=True)
    if isinstance(seed, int):
        seed = [seed]

    context_obs = torch.linspace(-3.0, -1.0, steps=design_dim).unsqueeze(-1).to(device)
    context_test = -context_obs

    annealing_frequency = 1000  #
    logging_frequency = 10  # log every 10th
    temp_annealing_frequency = num_steps // 5
    temp_gamma = 0.5

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
            "temp_annealing_frequency": temp_annealing_frequency,
            "temp_gamma": temp_gamma,
        }
    )

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
            tau=tau,
            hard=hard,
            temp_annealing_frequency=temp_annealing_frequency,
            temp_gamma=temp_gamma,
            run_id=mlflow.active_run().info.run_id,  # type: ignore
            experiment_id=mlflow.get_experiment_by_name(
                "Illustrative: Discrete Actions"
            ).experiment_id,  # type: ignore
        )
        for j, seed in enumerate(seed)
    )

    assert results is not None, "Error. Results is None"

    context_obs_df = pd.DataFrame(
        [res.get("context_obs").cpu().squeeze(-1).numpy() for res in results],
        index=seed,
        columns=[f"context_obs{i}" for i in range(1, context_obs.shape[0] + 1)],
    )
    context_test_df = pd.DataFrame(
        [res.get("context_test").squeeze(-1).cpu().numpy() for res in results],
        index=seed,
        columns=[f"context_test{i}" for i in range(1, context_test.shape[0] + 1)],
    )
    design_df = pd.DataFrame(
        [res.get("design").cpu().max(-1).indices.numpy() for res in results],
        index=seed,
        columns=[f"design{i}" for i in range(1, context_obs.shape[0] + 1)],
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

    mlflow.log_metric("mi_across_seeds", df["mi_mean"].mean(skipna=True))

    if num_true_models_to_sample == 0:
        df.to_csv(f"outputs/logging/results_table.csv")
        mlflow.log_artifact(f"outputs/logging/results_table.csv")
        return df

    # REGRETS
    # make sure everything is on CPU!!
    eval_results = Parallel(n_jobs=num_jobs, backend="loky")(
        delayed(calculate_regret)(
            context_obs=res.get("context_obs").cpu(),
            design=res.get("design").cpu(),
            context_test=res.get("context_test").cpu(),
            seed=seed[i],
            num_true_models_to_sample=num_true_models_to_sample,
            inference="snis",
        )
        for i, res in enumerate(results)
    )
    with open(f"outputs/logging/eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)
    mlflow.log_artifact(f"outputs/logging/eval_results.pkl")

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
    parser = argparse.ArgumentParser(description="Discrete actions")
    parser.add_argument("-o", type=str)

    parser.add_argument("--seed-reps", default=3, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--encoding-dim", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--num-jobs", default=-1, type=int)
    parser.add_argument("--num-steps", default=200, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--ucb-baseline", default=None, type=float)
    parser.add_argument("--thompson-sampling-baseline", action="store_true")
    parser.add_argument("--num-true-models-to-sample", default=3, type=int)

    parser.add_argument("--tau", default=3.0, type=float)
    parser.add_argument("--design-dim", default=5, type=int)
    parser.add_argument("--optimise-design", action="store_true")
    parser.add_argument("--hard", action="store_true")

    args = parser.parse_args()

    seed = [int(torch.rand(tuple()) * 2**30) for _ in range(args.seed_reps)]

    main(
        seed=seed,
        device=args.device,
        design_dim=args.design_dim,
        num_jobs=args.num_jobs,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        optimise_design=args.optimise_design,
        num_steps=args.num_steps,
        ucb_baseline=args.ucb_baseline,
        thompson_sampling_baseline=args.thompson_sampling_baseline,
        num_true_models_to_sample=args.num_true_models_to_sample,
        tau=args.tau,
        hard=args.hard,
    )
