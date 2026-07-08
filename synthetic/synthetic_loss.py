"""Synthetic LoRA vs. BaLoRA experiment (paper Figure 3).

Fine-tunes a small linear network on a random target with LoRA and BaLoRA
(balanced projection) and plots the loss over iterations, as the median +
inter-quartile band over ``num_trajectories`` seeds.

Key knobs (set below):
  - ``shapes``  : layer shapes; ``[(n, m)]`` is the one-layer panel, append a
                  second shape (e.g. ``[(n, m), (m, m)]``) for the two-layer panel.
  - ``right_scaling`` : initialization scaling (alpha / r); 1e0 in the paper.
  - ``adam``    : True to optimize with Adam, False for gradient descent.

Run from this directory (writes to figures/):
    python synthetic_loss.py
"""

import argparse
import os
import jax
import jax.numpy as jnp

import numpy as np

import copy
import itertools
import functools

import matplotlib.pyplot as plt

from layers import (
    get_trainable_parameters_lora,
    apply_linear,
    apply_lora_network,
    init_linear_network,
    init_lora_network,
)

from train import full_training, optimal_loss, project_balanced

from utils import (
    hessian_from_parameters,
    maximal_eigenvalue,
    canonical_minimizer,
    loss_hessian,
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 10.5})

### initialization and params

# target and lora rank
# n, m, r = 10, 5, 2
n, m, r = 20, 20, 4
shapes = [(n, m),]
L = len(shapes)
target_shape = (shapes[-1][0], shapes[0][1])

key = jax.random.PRNGKey(44)  # 44, 88
target = jax.random.normal(key, target_shape)

cropped = False  # only for L = 1
U, s, V = np.linalg.svd(target, full_matrices=False)
if cropped:
    s[r:] = 0
    target = U @ np.diag(s) @ V
print("Singular values of target:", s)

# model
mode = "random"  # "zero" or "random"
key, _ = jax.random.split(key)
init_frozen_params = init_linear_network(key, shapes, mode=mode)
pretrained = apply_linear(params=init_frozen_params)
print("mode for pretrained:", mode)

lora_rank = r
init_type = "random"   # "random", "unbalanced", "orthogonal"
right_scaling = 1e0
left_scaling = 0
init_lora_key = jax.random.PRNGKey(57) #np.random.randint(1000)  # 23 for divergence of theoretical rate

# optimization
n_lora_passes = 1
update_period = 100
total_n_steps = n_lora_passes * update_period
store_checkpoints = True
freq_checkpoints = 1  # total_n_steps // 100
adam = True

lora_learning_rate = 5e-3
select_best_learning_rate = True
grid = jnp.linspace(1e-5, 3e-1, 50) # use for L > 1
# grid = jnp.linspace(5e-3, 5e-1, 20) # use for L = 1
criterion = "final"  # "final_79", "threshold_1e-3"...

num_trajectories = 8

# plotting
remove_optimal_loss = True

### training

output_dict = {}

trajectory_types = ["BaLoRA", "LoRA"]
# trajectory_types = ["balanced", "regular", "new_balanced"]

for i in range(num_trajectories):
    for trajectory_type in trajectory_types:
        if trajectory_type == "BaLoRA":
            project_on_balanced = "old"
        # elif trajectory_type == "new_balanced":
        #     project_on_balanced = "new"
        elif trajectory_type == "LoRA":
            project_on_balanced = False

        init_params = init_lora_network(
            init_lora_key,
            shapes,
            lora_rank,
            init_mode="cola",
            frozen_params=init_frozen_params,
            right_scaling=right_scaling,
            left_scaling=left_scaling,
            init_type=init_type,
        )
        trainable_params, frozen_params = get_trainable_parameters_lora(
            copy.deepcopy(init_params)
        )

        init_lora_key, subkey = jax.random.split(init_lora_key)

        (
            new_trainable_params,
            new_frozen_params,
            losses,
            frozen_params_list,
            checkpoints,
            limiting_points,
            learning_rate_list,
        ) = full_training(
            "lora",
            total_n_steps,
            trainable_params,
            frozen_params,
            target,
            subkey,
            shapes,
            lora_rank,
            learning_rate=lora_learning_rate,
            project_on_balanced=project_on_balanced,
            select_best_learning_rate=select_best_learning_rate,
            grid=grid,
            store_checkpoints=store_checkpoints,
            freq_checkpoints=freq_checkpoints,
            random_init=True,
            verbose=False,
            store_limiting_points=True,
            right_scaling=right_scaling,
            left_scaling=left_scaling,
            criterion=criterion,
            init_type=init_type,
            adam=adam,
        )

        if len(losses) == 0:  # did not converge properly
            continue

        output_dict[trajectory_type + f"_{i}_losses"] = losses
        output_dict[trajectory_type + f"_{i}_frozen_params_list"] = frozen_params_list
        output_dict[trajectory_type + f"_{i}_checkpoints"] = checkpoints
        output_dict[trajectory_type + f"_{i}_limiting_points"] = limiting_points
        output_dict[trajectory_type + f"_{i}_learning_rate_list"] = learning_rate_list


### plotting the balancing gaps
# colormaps = {"LoRA": plt.get_cmap("winter", L), "BaLoRA": plt.get_cmap("autumn", L), "new_balanced": plt.get_cmap("spring", L)}

# plt.figure(figsize=(5, 3))
# x_axis = np.arange(total_n_steps)
# for trajectory_type in trajectory_types:
#     for l in range(L):
#         balancing_gap_list = []
#         for i in range(num_trajectories):
#             try:
#                 lora_checkpoints = output_dict[f"{trajectory_type}_{i}_checkpoints"]
#             except:
#                 continue
#             matrix_checkpoints = [(checkpoint["left"][f"layer_{l}"]["weight"], checkpoint["right"][f"layer_{l}"]["weight"].T) for checkpoint in lora_checkpoints]
#             balancing_gaps = np.array(
#                 [np.linalg.norm(U.T @ U - V.T @ V) for (U, V) in matrix_checkpoints]
#             )
#             balancing_gap_list.append(balancing_gaps)
#             plt.plot(x_axis, balancing_gaps, alpha=1 if num_trajectories == 1 else 0.5, color=colormaps[trajectory_type](l), label=f"{trajectory_type}" if l == 0 and i == 0 else None)
#         quantiles = np.quantile(
#             np.array(balancing_gap_list), [0.25, 0.5, 0.75], axis=0
#         )
#         plt.plot(x_axis, quantiles[1], label=f"median layer {l}", linestyle="--", color=colormaps[trajectory_type](l), linewidth=1)
#     # plt.fill_between(
#     #     x_axis,
#     #     quantiles[0],
#     #     quantiles[2],
#     #     alpha=0.2,
#     #     color=colormap(l),
#     # )
# plt.xlabel("Iteration")
# plt.ylabel("Balancing gap")
# plt.tight_layout()
# plt.legend()
# plt.grid()
# plt.savefig(f"figures/regular_balancing_gaps.pdf")

### plotting the loss

plt.figure(figsize=(2.5,2.2))
opt_loss = optimal_loss(target, pretrained, L, lora_rank)
print("Optimal loss LoRA:", opt_loss)
results = {}
for trajectory_type in trajectory_types:
        results[trajectory_type] = []

for i in range(num_trajectories):
    x_axis = np.arange(total_n_steps)
    for trajectory_type in trajectory_types:
        try:
            lora_losses = output_dict[trajectory_type + f"_{i}_losses"]
        except:
            continue

        if remove_optimal_loss:
            lora_losses = jnp.array(lora_losses) - opt_loss
        results[trajectory_type].append(np.array(lora_losses))
        plt.semilogy(x_axis, lora_losses, alpha=1 if num_trajectories == 1 else 0.3, linewidth=1, color="orange" if trajectory_type == "LoRA" else "cornflowerblue")

        if not remove_optimal_loss:
            plt.axhline(
                opt_loss,
                linestyle="--",
                label="Lora optimum",
            )

for trajectory_type in trajectory_types:
    quantiles = np.quantile(
        np.array(results[trajectory_type]), [0.25, 0.5, 0.75], axis=0
    )
    plt.semilogy(x_axis, quantiles[1], label=f"{trajectory_type}", linestyle="--", color="red" if trajectory_type == "LoRA" else "blue", linewidth=1)
    plt.fill_between(
        x_axis,
        quantiles[0],
        quantiles[2],
        alpha=0.2,
        color="orange" if trajectory_type == "LoRA" else "cornflowerblue",
    )
plt.xlabel("Iteration", fontsize=13)
# plt.ylabel("Loss")
plt.legend(handlelength=1.3)
plt.grid()
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig(f"figures/synthetic_loss_adam_{adam}.pdf")

