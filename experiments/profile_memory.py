#!/usr/bin/env python3
"""
Profile peak GPU memory for LoRA, BaLoRA, and DoRA during training of Llama-3.2-3B.

Loads the model and dataset exactly as train.py does (from $DSDIR and $SCRATCH).
Run from the repository root so the top-level modules are importable:

    python experiments/profile_memory.py [--steps 10] [--batch_size 1] [--seq_len 128] [--rank 8]
"""

import os
import sys
import gc
import argparse
from functools import partial

# Make the top-level modules (callbacks, peft_ga, dataset_utilities) importable
# when this script is launched from experiments/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from dataset_utilities.gsm8k_utilities import preprocess_batch
from dataset_utilities.common_utilities import DataCollatorForCausalLM
from peft_ga import LoraGAConfig, get_peft_model
from peft_ga.tuners.lora import LoraConfig
from peft_ga.utils.lora_ga_utils import estimate_gradient, LoraGAContext
from callbacks import ProjectionCallback

LORAGA_BSZ = 5    # batch size for LoRA-GA gradient estimation
LORAGA_ITERS = 4  # kept small so profiling stays fast (vs default 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10, help="Training steps per method")
parser.add_argument("--runs", type=int, default=3, help="Number of runs to average over")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--rank", type=int, default=8)
parser.add_argument(
    "--target_modules",
    nargs="+",
    default=["mlp.c_fc", "mlp.c_proj"],
    help="LoRA target modules (fewer = faster profiling)",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="gpt2",
)
args = parser.parse_args()

MODEL_NAME = args.model_name
NUM_STEPS = args.steps
NUM_RUNS = args.runs
BATCH_SIZE = args.batch_size
SEQ_LEN = args.seq_len
LORA_RANK = args.rank
TARGET_MODULES = args.target_modules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_model():
    """Mirror train.py: load from $DSDIR, add <|pad|> if needed."""
    model_path = os.environ["DSDIR"] + "/HuggingFace_Models/" + MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def prepare_dataset(tokenizer, n_samples: int):
    """Mirror train.py: load from $SCRATCH/datasets/gsm8k and tokenize."""
    dataset = load_from_disk(os.path.join(os.environ["SCRATCH"], "datasets", "gsm8k"))

    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=min(32, n_samples), seed=539)

    train_ds = dataset["train"].select(range(n_samples))

    tokenize = partial(preprocess_batch, tokenizer=tokenizer, max_length=SEQ_LEN)
    train_data = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)

    return train_data


def gpu_memory_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_memory():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def cleanup(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_method(
    method_name: str,
    dataset,
    tokenizer,
    *,
    use_dora: bool = False,
    project_every: int = 0,
    init_lora_weights=True,
    is_loraga: bool = False,
) -> float:
    print(f"\n{'─' * 55}")
    print(f"  Method : {method_name}")
    print(f"{'─' * 55}")

    model, tokenizer = load_base_model()
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    if is_loraga:
        lora_config = LoraGAConfig(
            r=LORA_RANK,
            lora_alpha=8,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
            init_lora_weights=init_lora_weights,
            use_dora=use_dora,
            scale="stable",
            stable_gamma=1.0,
            bsz=LORAGA_BSZ,
            iters=LORAGA_ITERS,
        )
        temp_set = dataset.select(range(LORAGA_BSZ * LORAGA_ITERS))
        named_grad = estimate_gradient(
            model=model,
            dataloader=DataLoader(temp_set, batch_size=LORAGA_BSZ, collate_fn=data_collator, shuffle=False),
            accelerator=None,
            quant_flag=False,
        )
        with LoraGAContext(model=model, named_grad=named_grad):
            model = get_peft_model(model=model, peft_config=lora_config)
    else:
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=init_lora_weights,
            use_dora=use_dora,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"/tmp/profile_{method_name.lower().replace(' ', '_')}",
        max_steps=NUM_STEPS,
        per_device_train_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=1,
        logging_steps=NUM_STEPS,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        label_names=["labels"],
    )

    callbacks = []
    if project_every > 0:
        callbacks.append(
            ProjectionCallback(model, project_every=project_every, start_step=0)
        )

    reset_peak_memory()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()

    peak = gpu_memory_gb()
    print(f"\n  Peak GPU memory: {peak:.3f} GB")

    cleanup(model, trainer)
    return peak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Steps      : {NUM_STEPS}")
    print(f"Runs       : {NUM_RUNS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Seq len    : {SEQ_LEN}")
    print(f"LoRA rank  : {LORA_RANK}")
    print(f"Targets    : {TARGET_MODULES}")

    # Load tokenizer once for dataset preparation, then free the base model
    _, tokenizer = load_base_model()
    n_samples = max(64, BATCH_SIZE * NUM_STEPS + 4, LORAGA_BSZ * LORAGA_ITERS)
    dataset = prepare_dataset(tokenizer, n_samples=n_samples)
    cleanup(_)

    methods = [
        dict(method_name="DoRA",    use_dora=True,  project_every=0, init_lora_weights=True,      is_loraga=False),
        dict(method_name="BaLoRA",  use_dora=False, project_every=1, init_lora_weights=True,      is_loraga=False),
        dict(method_name="LoRA",    use_dora=False, project_every=0, init_lora_weights=True,      is_loraga=False),
        dict(method_name="OLoRA",   use_dora=False, project_every=0, init_lora_weights="olora",   is_loraga=False),
        dict(method_name="LoRA-GA", use_dora=False, project_every=0, init_lora_weights="lora_ga", is_loraga=True),
    ]

    # { method_name: [peak_run1, peak_run2, ...] }
    results: dict[str, list[float]] = {cfg["method_name"]: [] for cfg in methods}

    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'━' * 55}")
        print(f"  Run {run}/{NUM_RUNS}")
        print(f"{'━' * 55}")
        for cfg in methods:
            name = cfg["method_name"]
            peak = profile_method(dataset=dataset, tokenizer=tokenizer, **cfg)
            results[name].append(peak)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'═' * 58}")
    print(f"  Peak GPU memory summary  ({MODEL_NAME})")
    print(f"  rank={LORA_RANK}, batch={BATCH_SIZE}, seq_len={SEQ_LEN}, runs={NUM_RUNS}")
    print(f"{'═' * 58}")
    lora_mean = sum(results["LoRA"]) / NUM_RUNS
    header = f"  {'Method':<12} {'Mean (GB)':>10}  {'Std (GB)':>10}  {'vs LoRA':>10}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for name, peaks in results.items():
        mean = sum(peaks) / len(peaks)
        std = (sum((p - mean) ** 2 for p in peaks) / len(peaks)) ** 0.5
        delta_pct = (mean - lora_mean) / lora_mean * 100 if lora_mean else 0
        delta_str = f"{delta_pct:+.1f}%" if name != "LoRA" else "baseline"
        print(f"  {name:<12} {mean:>10.3f}  {std:>10.3f}  {delta_str:>10}")
    print(f"{'═' * 58}\n")


if __name__ == "__main__":
    main()
