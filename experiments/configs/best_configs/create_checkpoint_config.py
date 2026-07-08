# I enter a sweep name (that I have already run) and an epoch number.
# The program parses the per-run loss files under experiments/results/<sweep_name>/,
# reads the loss at the end of the given (fraction of) epoch for every
# (learning_rate, scaling) in that sweep's params.json, and takes the argmin.
# It then copies the winning config from the sweep's config/ folder into
# best_config_<sweep_name>_epoch_<epoch>.json, flipping save_model to True.

import numpy as np
import json
import glob
import traceback
import os


sweep_name = "sweep_23_lorarite"
epoch = 1

loss_dictionary = {}
vmax, vmin = 0., np.inf

sweep_with_grep = ["sweep_21", "sweep_22", "sweep_23"]

## fill in loss dictionary
## loss_dictionary[{sweep_name}_epoch_{epoch}_{trajectory}] = 2D array of losses of sweep after epoch {epoch}

# Repo root (this file lives at experiments/configs/best_configs/).
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Sweeps are grouped by model_dataset; locate this sweep's config dir.
matches = glob.glob(os.path.join(root, "experiments", "configs", "*", sweep_name))
if not matches:
    raise FileNotFoundError(f"Could not find {sweep_name} under experiments/configs/*/")
sweep_config_dir = matches[0]
results_dir = os.path.join(root, "experiments", "results", sweep_name)

with open(f"{sweep_config_dir}/params.json", "r") as f:
    params = json.load(f)

for i, lr in enumerate(params["learning_rates"]):
    for j, scale in enumerate(params["A_scalings"]):
        try:
            filename = f"{results_dir}/losses_lr_{lr if sweep_name not in sweep_with_grep else i}_scaling_{scale if sweep_name not in sweep_with_grep else j}.json"
            print(filename)
            with open(filename, "r") as f:
                losses = json.load(f)
            if f"{sweep_name}_epoch_{epoch}" not in loss_dictionary:
                loss_dictionary[f"{sweep_name}_epoch_{epoch}"] = np.nan * np.ones((params["n_scalings"], params["n_lrs"]))
                loss_dictionary[f"{sweep_name}_epoch_{epoch}"][j, i] = losses[f"loss_after_epoch_{epoch}"]
            else:
                loss_dictionary[f"{sweep_name}_epoch_{epoch}"][j, i] = losses[f"loss_after_epoch_{epoch}"]
        except Exception as e:
            print(f"Error for lr {lr} scaling {scale}, epoch {epoch}: {type(e).__name__}: {e}")
            traceback.print_exc()

loss_matrix = loss_dictionary[f"{sweep_name}_epoch_{epoch}"]

scale_num, lr_num = np.unravel_index(np.nanargmin(loss_matrix), loss_matrix.shape)

src = f"{sweep_config_dir}/config/config_lr_{lr_num}_scale_{scale_num}.json"
dst = f"{root}/experiments/configs/best_configs/best_config_{sweep_name}_epoch_{epoch}.json"

with open(src, "r") as f:
    config = json.load(f)

config["save_model"] = True
config["num_train_epochs"] = epoch
config["output_dir"] = sweep_name + "/best_config"
config["logging_steps"] = 100
config["eval_strategy"] = "steps"
config["eval_steps"] = 100

with open(dst, "w") as f:
    json.dump(config, f, indent=2)
