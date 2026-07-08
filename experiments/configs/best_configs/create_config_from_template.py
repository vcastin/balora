import argparse
import json
import os

# Example usage: python create_config_from_template.py --lora-variant "LoRA" --model-name "Qwen/Qwen2.5-3B"

def create_config(seed=0, shuffle_seed=0, to_append=""):
    parser = argparse.ArgumentParser(description="Generates a json config automatically.")
    
    parser.add_argument("--lora-variant", type=str, required=True, help="Name of LoRA variant")
    parser.add_argument("--model-name", type=str, required=True, help="Model name, e.g. 'Qwen/Qwen2.5-3B', 'meta-llama/Llama-3.2-3B'")
    # parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name, e.g. 'MetaMathQA', 'wikitext-2-raw-v1', 'gsm8k', 'openhermes', 'code-feedback', 'ai2_arc_arc-challenge', 'alpaca', 'openorca', 'wizardlm'")
    # parser.add_argument("--param_dictionary", type=str, required=True, help="Path to best params json file, e.g. best_params_qwen_metamath_epoch_1.0.json")

    args = parser.parse_args()

    assert args.lora_variant in ["LoRA", "BaLoRA", "OLoRA", "DoRA", "LoRA-GA"], "Invalid LoRA variant. Must be one of 'LoRA', 'BaLoRA', 'OLoRA', 'DoRA', 'LoRA-GA'."

    assert args.model_name in ["Qwen/Qwen2.5-3B", "meta-llama/Llama-3.2-3B"], "Invalid model name. Must be one of 'Qwen/Qwen2.5-3B', 'meta-llama/Llama-3.2-3B'."

    if args.lora_variant in ["LoRA", "BaLoRA", "OLoRA", "DoRA"]:
        lora_variant = "lora"
    else:
        lora_variant = "loraga"

    if args.lora_variant == "BaLoRA":
        project_every = 1
    else:
        project_every = 0

    if args.lora_variant == "OLoRA":
        init_lora_weights = "olora"
    else:
        init_lora_weights = True

    if args.lora_variant == "DoRA":
        use_dora = True
    else:
        use_dora = False

    if args.model_name == "Qwen/Qwen2.5-3B":
        short_model_name = "qwen"
    elif args.model_name == "meta-llama/Llama-3.2-3B":
        short_model_name = "llama"

    short_dataset_names = {
        # "MetaMathQA": "metamath",
        "wikitext-2-raw-v1": "wikitext",
        "gsm8k": "gsm8k",
        "openhermes": "openhermes",
        "code-feedback": "codefeedback",
        "ai2_arc_arc-challenge": "arc",
        "alpaca": "alpaca",
        "openorca": "openorca",
        "wizardlm": "wizardlm"
        }
    
    pivotal_dataset = "metamath"

    with open("dataset_epochs.json", "r") as f:
        dataset_epochs = json.load(f)

    for dataset_name in short_dataset_names.keys():
        if dataset_name in ["arc", "gsm8k"]:
            logging_steps = 1
        else:
            logging_steps = 100

        epoch = dataset_epochs[short_dataset_names[dataset_name]]
        path_to_param_dictionary = f"best_params_{short_model_name}_{pivotal_dataset}_epoch_{epoch}.json"
        
        with open(path_to_param_dictionary, "r") as f:
            best_params = json.load(f)

        config = {
            "select_dataset_size": False,
            "start_step": 0,
            "save_model": True,
            "lora_variant": lora_variant,
            "model_name": args.model_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "shuffle_seed": shuffle_seed,
            # "train_data_size": 100000,
            "max_length": 1024,
            "train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1,
            "logging_strategy": "steps",
            "logging_steps": logging_steps,
            "save_strategy": "no",
            # "eval_data_size": 1000,
            "eval_batch_size": 8,
            "eval_strategy": "steps",
            "project_every": project_every,
            "projection_callback": "old",
            "A_scaling": best_params[args.lora_variant][1],
            "lora_rank": 8,
            "target_modules": [
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            "learning_rate": best_params[args.lora_variant][0],
            "lr_scheduler_type": "constant",
            "output_dir": f"./{short_model_name}_{short_dataset_names[dataset_name]}_{args.lora_variant}{to_append}",
            "optimizer": "AdamW",
            "init_lora_weights": init_lora_weights,
            "orthogonal_init": False,
            "use_dora": use_dora
        }


        file_name = f"{short_model_name}_{short_dataset_names[dataset_name]}_{args.lora_variant}{to_append}.json"
        with open(file_name, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ File '{file_name}' successfully created")

if __name__ == "__main__":
    for (seed, shuffle_seed, to_append) in zip([8002, 610, 6199], [23, 25, 90], ["", "_bis", "_tris"]):
        create_config(seed, shuffle_seed, to_append)