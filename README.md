# Contextual Optimisation via Bayesian Experimental Design.

Accepted at ICML 2023.

Paper: https://arxiv.org/abs/2302.14015

## Installation
Install the basic requirements with conda/mamba, e.g. by running

    conda env create -f env.yml

Make sure you are installing the correct torch version (cpu vs gpu).


## Experiments

To reproduce the experiments in the paper follow the instructions below.

-------------
### **Experiment 1: Discrete treatments**
-------------
This is the example with 4 treatments.
**MODEL** Each of the four treatments $a = 1, 2, 3, 4$ is a random function with two parameters $\psi_k = (\psi_{k,1}, \psi_{k,2})$ with the following Gaussian priors (parameterized by mean and covariance matrix):

$$
    \begin{align}
    \psi_1 ~ &\sim \mathcal{N} \left(\begin{pmatrix} 5.00 \\ 15.0  \end{pmatrix},
    \begin{pmatrix}
     9.00 & 0 \\
     0 & 9.00
    \end{pmatrix} \right) \\
    \psi_2 ~ &\sim \mathcal{N} \left(\begin{pmatrix} 5.00 \\ 15.0  \end{pmatrix},
    \begin{pmatrix}
     2.25 & 0 \\
     0 & 2.25
    \end{pmatrix} \right) \\
    \psi_3 ~ &\sim \mathcal{N} \left(\begin{pmatrix} -2.0 \\ -1.0  \end{pmatrix},
    \begin{pmatrix}
     1.21 & 0 \\
     0 & 1.21
    \end{pmatrix} \right) \\
    \psi_4 ~ &\sim \mathcal{N} \left(\begin{pmatrix} -7.0 \\ 3.0  \end{pmatrix},
    \begin{pmatrix}
     1.21 & 0 \\
     0 & 1.21
    \end{pmatrix} \right)
    \end{align}
$$

and reward (outcome) likelihoods:

$$
    \begin{align}
    y | c, a, \psi &\sim  \mathcal{N} \left( f(c, a, \psi), 0.1 \right) \\
    f(c, a, \psi) &= -c^2 + \beta(a, \psi)c + \gamma(a, \psi) \\
    \gamma &= (\psi_{a, 1} + \psi_{a, 2} + 18) / 2 \\
    \beta &= (\psi_{a, 2} - \gamma + 9) / 3
    \end{align}
$$


**Run an experiment**

The model is implemented as `ContinuousContextDiscreteTreatment` class, defined in `bayesian_simulators/continuous_scalar_context_discrete_treatment.py`.


The training loop is implemented in
**`research_experiments/discrete_design_main.py`**. In addition to training loop (`train_net`), that script also:
- defines a function to compute UCB designs: `compute_ucb_design`;
- defines a function to calculate regret: `calculate_regret`. The "real process" is `ContinuousContextAndTreatment` with a fixed realisation of the parameters $\psi$, drawn from the prior. It differentiates the REAL WORLD SIMULATOR from the model emulator by denoting the former in ALL CAPS;
- defines `main` which defines a prior, sets up logging, runs all the above (train loop, regret evaluation), stores the outputs and creates some plots.

To run CO-BED:

```
python discrete_design_main.py \
    --seed-reps 3 \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --tau 5 \
    --optimise-design \
    --device <cpu/cuda>
```

To run random baseline, simply remove the `optimise-design` flag:

```
python discrete_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --tau 5 \
    --device <cpu/cuda>
```

To run UCB baseline, use `ucb-baseline` flag, setting to the appropriate level of $\alpha$, e.g. 0.0 (or $1.0$ or $2.0$):

```
python discrete_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --tau 5 \
    --ucb-baseline 0.0 \
    --device <cpu/cuda>
```

To run Thompson sampling baseling, use `thompson-sampling-baseline` flag:
```
python discrete_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --tau 5 \
    --thompson-sampling-baseline \
    --device <cpu/cuda>
```

----------------
### **Experiment 2: Continuous treatments**
--------------
**MODEL** For the continuous treatment example we use the following model:

$$
 \text{Prior: }  \quad  \psi = (\psi_1, \psi_2, \psi_3, \psi_4 ), \quad  \psi_i \sim \text{Uniform}[0.1, 1.1] \; \text{iid}
$$
$$
 \text{Likelihood: } \quad y  | c, a, \psi \sim \mathcal{N} (f(\psi, a, c), \sigma^2),
$$
where
$$
    f(\psi, a, c) = \exp\left( -\frac{\big(a - g(\psi, c)\big)^2}{h(\psi, c)} \right) \quad
    g(\psi, c) = \psi_0 + \psi_1 c + \psi_2 c^2 \quad
    h(\psi, c) = \psi_3
$$



**Run an experiment**

The model is implemented as `ContinuousContextAndTreatment` class, defined in `bayesian_simulators/continuous_scalar_context_scalar_treatment.py`.

The training loop is implemented in
**`research_experiments/continuous_design_main.py`**. In addition to training loop (`train_net`), that script also:
- defines a function to compute UCB designs: `compute_ucb_design`;
- defines a function to calculate regret: `calculate_regret`. The "real process" is `ContinuousContextAndTreatment` with a fixed realisation of the parameters $\psi$, drawn from the prior.It differentiates the REAL WORLD SIMULATOR from the model emulator by denoting the former in ALL CAPS;
- defines `main` which defines a prior, sets up logging, runs all the above (train loop, regret evaluation), stores the outputs and creates some plots.

To run CO-BED:

```
python continuous_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --optimise-design \
    --device <cpu/cuda>
```

To run random baseline, simply remove the `optimise-design` flag:

```
python continuous_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --device <cpu/cuda>
```

To run UCB baseline, use `ucb-baseline` flag, setting to the appropriate level of $\alpha$, e.g. 0.0 (or $1.0$ or $2.0$):

```
python continuous_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --ucb-baseline 0.0 \
    --device <cpu/cuda>
```

To run Thompson sampling baseling, use `thompson-sampling-baseline` flag:
```
python continuous_design_main.py \
    --batch-size 512 \
    --hidden-dim 512 \
    --encoding-dim 16 \
    --lr 0.001 \
    --gamma 0.9 \
    --num-jobs -1 \
    --num-steps 500 \
    --design-dim 10 \
    --seed-reps 5 \
    --num-true-models-to-sample 3 \
    --thompson-sampling-baseline \
    --device <cpu/cuda>
```

If you want to learn a batch of $D$, e.g. `D=60` experiments modify `design_dim` accordingly. e.g:
```
python
design_dim: 60
```


# Logging

Logging is done with MlFlow, to start a local server, run

    mlflow ui

and navigate to the appropriate experiment to view all logged metrics.


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
