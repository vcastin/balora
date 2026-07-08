"""Config-driven fine-tuning entrypoint for BaLoRA and the LoRA variants.

Runs a single (learning-rate, scaling) point of a sweep from a JSON config:

    python train.py --config experiments/configs/llama_wikitext/sweep_22/config/config_lr_0_scale_0.json

The LoRA variant is selected by the config fields ``lora_variant``, ``project_every``
(BaLoRA when > 0), ``use_dora`` (DoRA), ``orthogonal_init`` / ``init_lora_weights``
(OLoRA) and ``optimizer`` (``lora_rite``). See the README for the full mapping.
"""

import torch
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from peft import AdaLoraConfig, LoraConfig, TaskType
from peft_ga import LoraGAConfig, get_peft_model
from peft_ga.utils.lora_ga_utils import estimate_gradient, LoraGAContext, save_loraga_model_init, save_loraga_model_final
from datasets import load_from_disk, DatasetDict
from callbacks import BalancingGapCallback, BestProjectionCallback, ProjectionCallback, SaveValLossCallback
from reflora import Refactorer, RefTrainer
from lora_inits import orthogonal_init_lora
import os
import json
import argparse


def parse_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

parser = argparse.ArgumentParser(description="Finetuning config")
parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
args = parser.parse_args()

config = parse_config(args.config)

torch.manual_seed(config["seed"])

lora_rank = config["lora_rank"]
output_dir = config["output_dir"]
max_length = config["max_length"]
select_dataset_size = config["select_dataset_size"]  # True or False


## Model and Tokenizer
root = os.environ['DSDIR'] + '/HuggingFace_Models'
model_name = config["model_name"]

model = AutoModelForCausalLM.from_pretrained(root + '/' + model_name,
                                              device_map="auto",
                                            )
tokenizer = AutoTokenizer.from_pretrained(root + '/' + model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))

## Load dataset
root = os.environ['SCRATCH']
dataset = load_from_disk(os.path.join(root, "datasets", config["dataset_name"]))


# Helper function to load instruction-tuned tokenizer
def load_instruct_tokenizer(model_name):
    """Load instruction-tuned tokenizer and set pad_token if needed."""
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ['DSDIR'] + '/HuggingFace_Models/' + model_name + "-Instruct",
        use_fast=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_instruction_dataset(dataset_name, dataset, config):
    """Setup preprocessing for instruction-following datasets using common utilities."""
    # Import the appropriate preprocessing module
    if dataset_name == "code-feedback":
        from dataset_utilities.code_feedback_utilities import preprocess_batch
    elif dataset_name == "openhermes":
        from dataset_utilities.openhermes_utilities import preprocess_batch
    elif dataset_name == "wizardlm":
        from dataset_utilities.wizardlm_utilities import preprocess_batch
    elif dataset_name == "openorca":
        from dataset_utilities.openorca_utilities import preprocess_batch
    elif dataset_name == "alpaca":
        from dataset_utilities.alpaca_utilities import preprocess_batch
    elif dataset_name == "ai2_arc_arc-challenge":
        from dataset_utilities.ai2_arc_utilities import preprocess_batch
    else:
        raise ValueError(f"Unknown instruction dataset: {dataset_name}")

    from dataset_utilities.common_utilities import DataCollatorForCausalLM

    # Load instruction-tuned tokenizer
    tokenizer = load_instruct_tokenizer(config["model_name"])

    # Setup preprocessing
    tokenize = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        max_length=config["max_length"]
    )

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    return tokenizer, tokenize, data_collator, dataset


# Datasets that use instruction-tuned tokenizer and common utilities
INSTRUCTION_DATASETS = {
    "code-feedback", "openhermes", "wizardlm", "openorca", "alpaca", "ai2_arc_arc-challenge"
}

# Setup dataset-specific preprocessing
dataset_name = config["dataset_name"]

if dataset_name == "gsm8k":
    from dataset_utilities.gsm8k_utilities import preprocess_batch, DataCollatorForCausalLM
    tokenize = partial(preprocess_batch, tokenizer=tokenizer, max_length=config["max_length"])
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

elif dataset_name == "MetaMathQA":
    from dataset_utilities.metamathQA_utilities import preprocess_batch, DataCollatorForCausalLM
    tokenize = partial(preprocess_batch, tokenizer=tokenizer, max_length=config["max_length"])
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

elif dataset_name in INSTRUCTION_DATASETS:
    # # Handle special dataset preparation for code-feedback
    # if dataset_name == "code-feedback" and isinstance(dataset, DatasetDict):
    #     # Rename the fields as "train" and "test"
    #     dataset = DatasetDict({
    #         "train": dataset["train_sft"],
    #         "test": dataset["test_sft"],
    #     })

    # Use common setup for all instruction datasets
    tokenizer, tokenize, data_collator, dataset = setup_instruction_dataset(
        dataset_name, dataset, config
    )

else:
    # Default fallback for generic text datasets
    def tokenize(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=config["eval_data_size"], seed=config["shuffle_seed"])

if select_dataset_size:
    train_ds = (dataset["train"].select(range(config["train_data_size"]))).shuffle(seed=config["shuffle_seed"])
    test_ds = (dataset["test"].select(range(config["eval_data_size"]))).shuffle(seed=config["shuffle_seed"])
else:
    train_ds = dataset["train"].shuffle(seed=config["shuffle_seed"])
    test_ds = dataset["test"].shuffle(seed=config["shuffle_seed"])

train_data = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
eval_data = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)

# checking how many samples have been discarded (too long seq length)
final_len_train = len(train_data)
final_len_eval = len(eval_data)
if select_dataset_size:
    print("Percentage of discarded samples in train set:", (config["train_data_size"] - final_len_train) / config["train_data_size"] * 100)
    print("Percentage of discarded samples in eval set:", (config["eval_data_size"] - final_len_eval) / config["eval_data_size"] * 100)


dataloader = DataLoader(
    train_data,
    batch_size=config["train_batch_size"],
    shuffle=False,
    collate_fn=data_collator,
)


## LoRA Configuration
# "reflora" uses standard LoRA initialization; the Refactorer is set up later
effective_lora_variant = "lora" if config["lora_variant"] == "reflora" else config["lora_variant"]

if effective_lora_variant == "adalora":
    lora_config = AdaLoraConfig(
        r=lora_rank,
        lora_alpha=8,
        target_modules=config["target_modules"],
        lora_dropout=0.0,  # in submission: 0.05
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        init_lora_weights=config["init_lora_weights"],
        use_dora=config["use_dora"],
        target_r=lora_rank,  # default init_r
    )
elif effective_lora_variant == "loraga":
    lora_config = LoraGAConfig(
        r=lora_rank,
        lora_alpha=8,
        target_modules=config["target_modules"],
        lora_dropout=0.0,  # in submission: 0.05
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        init_lora_weights=config["init_lora_weights"],
        use_dora=config["use_dora"],
        scale="stable",
        stable_gamma=1/config["A_scaling"] ** 2,
        bsz=config.get("loraga_bsz", 2),      # gradient-estimation batch size (default 2)
        iters=config.get("loraga_iters", 64),  # gradient-estimation iterations (default 64)
    )

    temp_set = train_data.select(range(lora_config.bsz * lora_config.iters))

    named_grad = estimate_gradient(
        model=model,
        dataloader=DataLoader(temp_set, batch_size=lora_config.bsz, collate_fn=data_collator, shuffle=False),
        accelerator=None,
        quant_flag=False,
    )

    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=lora_config)
    
    save_loraga_model_init(model, save_dir=output_dir)

else:
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=8 * config["A_scaling"] ** 2,
        target_modules=config["target_modules"],
        lora_dropout=0.0,  # in submission: 0.05
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        init_lora_weights=config["init_lora_weights"],
        use_dora=config["use_dora"],
    )
    model = get_peft_model(model, lora_config)

if config["orthogonal_init"]:
    model.apply(orthogonal_init_lora)

if effective_lora_variant == "adalora":
    for name, module in model.named_modules():
        if hasattr(module, "lora_E"):
            with torch.no_grad():
                module.lora_E.default.weight *= config["A_scaling"]


global_dir = os.path.basename(os.path.dirname(output_dir))  # this is sweep_n for a sweep

## Training Arguments
training_args = TrainingArguments(
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    output_dir=global_dir,
    eval_strategy=config["eval_strategy"],
    save_strategy=config["save_strategy"],
    num_train_epochs=config["num_train_epochs"],
    logging_dir=f"./logs/{output_dir}",
    logging_strategy=config["logging_strategy"],
    logging_steps=config["logging_steps"],
    learning_rate=config["learning_rate"],
    lr_scheduler_type=config["lr_scheduler_type"],
    per_device_train_batch_size=config["train_batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    label_names=["labels"],
    report_to="tensorboard",
    # remove_unused_columns=False,
    # eval_accumulation_steps=5,
)


## Train the model

# RefLoRA uses the Trainer's internally-created optimizer (hooks are attached via create_optimizer).
# All other variants use an externally-created optimizer passed to the Trainer.
refactorer = None
optimizer = None

if config["lora_variant"] == "reflora":
    refactorer = Refactorer(
        model,
        use_scalar=config.get("reflora_use_scalar", False),
        warmup_steps=config.get("reflora_warmup_steps", 100),
        interval=config.get("reflora_interval", 1),
        re_init=True,
    )
elif config["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "lora_rite":
    from lora_rite import LoRARite
    from collections import defaultdict
    layer_params = defaultdict(dict)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_A' in name:
            key = name.split('.lora_A.')[0]
            layer_params[key]['A'] = param
        elif 'lora_B' in name:
            key = name.split('.lora_B.')[0]
            layer_params[key]['B'] = param
    lora_params = []
    for key in sorted(layer_params.keys()):
        if 'A' in layer_params[key] and 'B' in layer_params[key]:
            lora_params.append(layer_params[key]['A'])
            lora_params.append(layer_params[key]['B'])
    optimizer = LoRARite(lora_params, lr=config["learning_rate"], betas=(0.9, 0.999))

TrainerClass = RefTrainer if refactorer is not None else Trainer
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
if optimizer is not None:
    trainer_kwargs["optimizers"] = (optimizer, None)
if refactorer is not None:
    trainer_kwargs["refactorer"] = refactorer

trainer = TrainerClass(**trainer_kwargs)

## uncomment for storing the balancing gap and norms
# callback = BalancingGapCallback(model, balancing_gap_file=f"./logs/{output_dir}/balancing_gap.json", norm_file=f"./logs/{output_dir}/norms.json")
# trainer.add_callback(callback)

output_file = global_dir + f'/losses_lr_{config["learning_rate"]}_scaling_{config["A_scaling"]}.json'
save_loss_callback = SaveValLossCallback(output_file, trainer)
trainer.add_callback(save_loss_callback)

if config["project_every"] > 0:
    if config["projection_callback"] == "best":
        projection_callback = BestProjectionCallback(model, config["project_every"], config["start_step"])
    else:
        projection_callback = ProjectionCallback(model, config["project_every"], config["start_step"])
    trainer.add_callback(projection_callback)

trainer.train()

if config["save_model"]:
    if effective_lora_variant == "loraga":
        save_loraga_model_final(model, save_dir=output_dir)
    else:
        folder_name = output_dir + "/"
        trainer.model.save_pretrained(folder_name)
    tokenizer.save_pretrained(output_dir)




