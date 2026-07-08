import numpy as np
import json
import os

if not os.path.exists("config"):
    os.makedirs("config")

n_lrs = 10
n_scalings = 5
learning_rates = np.logspace(-5., -2., num=n_lrs)
A_scalings = np.logspace(-2., 2., num=n_scalings)

# save in json file
import os
with open("params.json", "w") as f:
    json.dump({
        "n_lrs": n_lrs,
        "n_scalings": n_scalings,
        "learning_rates": learning_rates.tolist(),
        "A_scalings": A_scalings.tolist()
    }, f, indent=2)

for i, learning_rate in enumerate(learning_rates):
    for j, A_scaling in enumerate(A_scalings):
        config = {
          "select_dataset_size": False,
          "start_step": 0,
          "save_model": False,
          "lora_variant": "lora",
          "model_name": "meta-llama/Llama-3.2-3B",
          "dataset_name": "gsm8k",
          "seed": 8002,
          "shuffle_seed": 33,
          "train_data_size": 30000,
          "max_length": 1024,
          "train_batch_size": 8,
          "gradient_accumulation_steps": 4,
          "num_train_epochs": 3,
          "logging_strategy": "steps",
          "logging_steps": 100,
          "save_strategy": "no",
          "eval_data_size": 1000,
          "eval_batch_size": 8,
          "eval_strategy": "epoch",
          "project_every": 0,
          "projection_callback": "old",
          "A_scaling": A_scaling,
          "lora_rank": 8,
          "target_modules": ["gate_proj", "down_proj", "up_proj"],
          "learning_rate": learning_rate,
          "lr_scheduler_type": "constant",
          "output_dir": f"./sweep_31/lr_" + str(learning_rate) + "_scale_" + str(A_scaling),
          "optimizer": "AdamW",
          "init_lora_weights": True,
          "orthogonal_init": False,
          "use_dora": True,
        }

        with open(f"config/config_lr_{i}_scale_{j}.json", "w") as f:
            json.dump(config, f, indent=2)