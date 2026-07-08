import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter

import json
import os
import glob
import argparse

if not os.path.exists("figures"):
    os.makedirs("figures")

# Example usage: python plot_sweeps.py --config 'plot_configs/qwen_metamath.json'
#
# Run from experiments/plotting/. Sweep configs are grouped by model_dataset
# under experiments/configs/<group>/sweep_N/; per-run loss files are expected
# under experiments/results/sweep_N/ (see README).

CONFIGS_ROOT = os.path.join("..", "configs")
RESULTS_ROOT = os.path.join("..", "results")


def find_sweep_config_dir(sweep_number):
    """Locate a sweep's config dir regardless of which model_dataset group it lives in."""
    matches = glob.glob(os.path.join(CONFIGS_ROOT, "*", f"sweep_{sweep_number}"))
    if not matches:
        raise FileNotFoundError(f"sweep_{sweep_number} not found under {CONFIGS_ROOT}/*/")
    return matches[0]


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

cmaps = [plt.get_cmap("seismic")]
flattened_figsize = (7.3, 3.)
add_cmap_dots = False

def parse_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

parser = argparse.ArgumentParser(description="plot_config")
parser.add_argument('--config', type=str, required=True, help="Path to plot config file")
args = parser.parse_args()
config = parse_config(args.config)

n_rows = config["n_rows"]

loss_dictionary = {}
best_params_dict = {}
vmax, vmin = 0., np.inf

## fill in loss dictionary
## loss_dictionary[{method}_epoch_{epoch}_{trajectory}] = 2D array of average losses of sweep after epoch {epoch}

for epoch in config["epoch_list"]:
    print(f"Processing epoch {epoch}")

    for method in config["sweep_dictionary"].keys():
        first_sweep_number = config["sweep_dictionary"][method][0]
        with open(os.path.join(find_sweep_config_dir(first_sweep_number), "params.json"), "r") as f:
            params = json.load(f)

        shape = (params["n_scalings"], params["n_lrs"])
        sum_losses = np.zeros(shape)
        count_samples = np.zeros(shape)

        for sweep_number in config["sweep_dictionary"][method]:
            for i, lr in enumerate(params["learning_rates"]):
                for j, scale in enumerate(params["A_scalings"]):
                    try:
                        path = os.path.join(RESULTS_ROOT, f"sweep_{sweep_number}", f"losses_lr_{i}_scaling_{j}.json")
                        with open(path, "r") as f:
                            losses = json.load(f)
                        
                        sum_losses[j, i] += losses[f"epoch_{epoch}"]
                        count_samples[j, i] += 1
                    except:
                        pass

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_loss_matrix = sum_losses / count_samples
            avg_loss_matrix[count_samples == 0] = np.nan 

        loss_dictionary[f"{method}_epoch_{epoch}"] = avg_loss_matrix

        if not np.all(np.isnan(avg_loss_matrix)):
            vmax = max(vmax, np.nanmax(avg_loss_matrix))
            vmin = min(vmin, np.nanmin(avg_loss_matrix))
            idx = np.nanargmin(avg_loss_matrix)
            
            j_min, i_min = np.unravel_index(idx, avg_loss_matrix.shape)
            
            best_lr = params["learning_rates"][i_min]
            best_scaling = params["A_scalings"][j_min]
            
            best_params_dict[method] = [best_lr, best_scaling]
            
            print(f"Best for {method}: LR={best_lr}, Scaling={best_scaling} (Loss={avg_loss_matrix[j_min, i_min]:.6f})")
        else:
            print(f"No data available for {method} to find a minimum.")


    with open(os.path.join(CONFIGS_ROOT, "best_configs", f"best_params_{config['global_name']}_epoch_{epoch}.json"), "w") as f:
        json.dump(best_params_dict, f, indent=4)

    lrs = np.array(params["learning_rates"])
    scales = np.array(params["A_scalings"])
    LR_mesh, SCALE_mesh = np.meshgrid(lrs, scales)
    product_matrix = LR_mesh * SCALE_mesh

    # plotting grids of final losses
    figsize = (1.6 * len(config["sweep_dictionary"]) // n_rows + 0.27, 2. * n_rows)

    fig, axes = plt.subplots(n_rows, len(config["sweep_dictionary"]) // n_rows, figsize=figsize, sharey=True, constrained_layout=True)

    for i, method in enumerate(config["sweep_dictionary"].keys()):
        if len(config["sweep_dictionary"]) == 1:
            ax_i = axes
        elif n_rows == 1:
            ax_i = axes[i]
        elif n_rows > 1:
            ax_i = axes[i // (len(config["sweep_dictionary"]) // n_rows), i % (len(config["sweep_dictionary"]) // n_rows)]

        # load params of sweep
        first_sweep_number = config["sweep_dictionary"][method][0]
        with open(os.path.join(find_sweep_config_dir(first_sweep_number), "params.json"), "r") as f:
            params = json.load(f)

        loss_matrix = loss_dictionary[f"{method}_epoch_{epoch}"]

        lr_grid = np.logspace(np.log10(params["learning_rates"][0]) - 0.005, np.log10(params["learning_rates"][-1]) + 0.005, params["n_lrs"] + 1)
        init_grid = np.logspace(np.log10(params["A_scalings"][0]) - 0.005, np.log10(params["A_scalings"][-1]) + 0.005, params["n_scalings"] + 1)

        mesh = ax_i.pcolormesh(lr_grid, init_grid, loss_matrix, norm=colors.LogNorm(vmin=vmin,vmax=config["vmax_over_vmin"] * vmin), shading='auto')
        ax_i.set_xscale('log')
        ax_i.set_yscale('log')
        ax_i.set_title(f"{method}")
        if i // (len(config["sweep_dictionary"]) // n_rows) == (n_rows - 1):
            ax_i.set_xlabel('Learning Rate')
        if i % (len(config["sweep_dictionary"]) // n_rows) == 0:
            ax_i.set_ylabel('Right Scaling')

    if n_rows == 1:
        cbar = fig.colorbar(mesh, ax=[axes[k] for k in range(len(config["sweep_dictionary"]))] if len(config["sweep_dictionary"]) > 1 else axes)
    else:
        cbar = fig.colorbar(mesh, ax=[axes[r, c] for r in range(n_rows) for c in range(len(config["sweep_dictionary"]) // n_rows)])
    if config["colorbar_ticks"] is not None and config["colorbar_ticklabels"] is not None:
        cbar.set_ticks(config["colorbar_ticks"])
        cbar.set_ticklabels(config["colorbar_ticklabels"])

    plt.savefig('figures/' + config["global_name"] + f"_epoch_{epoch}.pdf")


    ## plotting the flattened losses
    fig, axes = plt.subplots(1, 2, figsize=flattened_figsize)
    for s, method in enumerate(config["sweep_dictionary"].keys()):
        # load params again
        first_sweep_number = config["sweep_dictionary"][method][0]
        with open(os.path.join(find_sweep_config_dir(first_sweep_number), "params.json"), "r") as f:
            params = json.load(f)
            
        loss_matrix  = loss_dictionary[f"{method}_epoch_{epoch}"]
        for a in range(2):
            try:
                min_along_axis = np.nanmin(loss_matrix, axis=a)
                min_indices = np.nanargmin(loss_matrix, axis=a)
                values = [params["learning_rates"][min_indices[l]] if a == 1 else params["A_scalings"][min_indices[l]] for l in range(len(min_indices))]
            except:
                print(f"Could not compute min along axis {a} for {method} at epoch {epoch}")
                loss_matrix = np.nan_to_num(loss_matrix, nan=10)
                min_along_axis = np.min(loss_matrix, axis=a)
                continue

            ax = axes[a]
            x_axis = params["learning_rates"] if a == 0 else params["A_scalings"]
            if config["plot_type"] == "plot":
                ax.plot(x_axis, min_along_axis, label=method, linewidth=1.1, alpha=0.9)
            else:
                ax.scatter(x_axis, min_along_axis, label=method, linewidth=1.1, alpha=0.9, s=5, color=cmaps[0]((s + 1)/(len(config["sweep_dictionary"].keys()) + 1)))
            if add_cmap_dots:
                sc = ax.scatter(x_axis,
                                min_along_axis, s=10,
                                c=values, cmap=cmaps[0], norm=colors.LogNorm(vmin=params["A_scalings"][0] if a == 0 else params["learning_rates"][0], vmax=params["A_scalings"][-1] if a == 0 else params["learning_rates"][-1]))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Right Scaling' if a == 1 else 'Learning Rate')
            ax.set_ylabel(f'Final loss')
            if config["ylim"] is not None:
                ax.set_ylim(config["ylim"][a][0], config["ylim"][a][1])
            if config["yticks"] is not None:
                ax.set_yticks(config["yticks"][a])
                ax.set_yticklabels(config["ytick_labels"][a])
                ax.yaxis.set_minor_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.ScalarFormatter())
                ax.yaxis.set_tick_params(which='minor', left=False, right=False)

            if s == 0 and add_cmap_dots:
                cbar = fig.colorbar(sc, ax=ax, label="best scaling" if a == 0 else "best learning rate")
        if config["xlim"][0] is not None:
            axes[0].set_xlim(config["xlim"][0][0], config["xlim"][0][1])
        if config["xlim"][1] is not None:
            axes[1].set_xlim(config["xlim"][1][0], config["xlim"][1][1])
        axes[1].legend(handlelength=1.)
    plt.tight_layout()
    plt.savefig('figures/' + config['global_name'] + f"_epoch_{epoch}_flattened.pdf")