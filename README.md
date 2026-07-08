# Balanced LoRA (BaLoRA)

Code for the paper **“Balanced LoRA: Removing Parameter Invariance to Accelerate
Convergence.”**

> Low-Rank Adaptation (LoRA) is inherently overparameterized: many pairs of
> low-rank factors `(A, B)` yield the same adapted matrix `AB`, but they have
> different condition numbers and therefore different convergence rates.
> **Balanced LoRA (BaLoRA)** projects the adapters onto the *balanced manifold*
> `{(A, B) : Aᵀ A = B Bᵀ}` after each optimizer step. This projection is cheap,
> leaves the product `AB` unchanged, and improves the conditioning of the loss —
> so BaLoRA converges faster than LoRA and matches or beats several LoRA variants.

The whole method is a lightweight post-optimizer projection; see
[`callbacks.py`](callbacks.py) (`ProjectionCallback`) and Algorithms 1–2 of the paper.

---

## Repository layout

```
balora/
├── train.py                  # config-driven fine-tuning entrypoint (all LoRA variants)
├── callbacks.py              # BaLoRA balanced-projection callback + val-loss logging
├── lora_inits.py             # orthogonal (OLoRA-style) initialization
├── lora_rite.py              # LoRA-RITE optimizer
├── peft_ga/                  # vendored, patched PEFT (adds LoRA-GA; used by every variant)
├── reflora/                  # RefLoRA (Refactorer + trainer)
├── dataset_utilities/        # per-dataset tokenization / collators
├── download_datasets.py      # download + cache all datasets to $SCRATCH/datasets/<name>
│
├── experiments/
│   ├── configs/              # sweep definitions, grouped by model_dataset (see below)
│   │   ├── llama_wikitext/
│   │   ├── llama_gsm8k/
│   │   ├── qwen_wikitext/
│   │   ├── qwen_metamath/
│   │   ├── qwen_openhermes/
│   │   └── best_configs/     # best-hyperparameter single runs (Table 1, Table 5, Fig. 10)
│   ├── profile_memory.py     # peak-GPU-memory comparison (Table 6)
│   └── plotting/             # plot_sweeps.py (heatmaps), runtime_loss_plot.py (loss vs time)
│
└── synthetic/                # JAX toy framework for the synthetic experiments (Fig. 3)
    ├── layers.py  train.py  utils.py
    └── synthetic_loss.py
```

Each `experiments/configs/<group>/sweep_N/` directory contains a `create_config.py`
(generates the learning-rate × scaling grid of JSON configs into a `config/`
subfolder) and an `execute_array.slurm` (the SLURM array job that ran it).

---

## Installation

Requires **Python ≥ 3.10** (the vendored `peft_ga` uses `match`/`case`).

```bash
pip install -r requirements.txt
```

`torch`/`transformers`/`peft`/`datasets` cover the LLM experiments; `jax`/`jaxlib`
cover the synthetic experiments.

## Data

`train.py` loads each dataset from `$SCRATCH/datasets/<dataset_name>`. Download and
cache them there with:

```bash
export SCRATCH=/path/to/cache
python download_datasets.py --datasets all        # or: wikitext gsm8k metamath openhermes ...
```

Base models are read from `$DSDIR/HuggingFace_Models/<model_name>` (the Jean Zay
convention used for the paper); point `$DSDIR` at any directory containing the HF
model snapshots, or adapt the two `from_pretrained` calls in `train.py` to load
directly from the hub.

---

## Running an experiment

A single point of a sweep is one JSON config:

```bash
python train.py --config experiments/configs/llama_wikitext/sweep_22/config/config_lr_0_scale_0.json
```

To (re)generate a whole grid and launch it on a cluster:

```bash
cd experiments/configs/llama_wikitext/sweep_22
python create_config.py                 # writes config/config_lr_*_scale_*.json + params.json
sbatch execute_array.slurm              # submit from the repo root; edit #SBATCH lines for your cluster
```

### Which config field selects which method

Every LoRA variant is one entrypoint (`train.py`) driven by these config fields:

| Method    | Selected by                                                            |
|-----------|------------------------------------------------------------------------|
| LoRA      | `lora_variant="lora"`, `project_every=0`                               |
| **BaLoRA**| `lora_variant="lora"`, `project_every=1` (`projection_callback="old"`) |
| DoRA      | `use_dora=true`                                                        |
| OLoRA     | `init_lora_weights="olora"`                                            |
| LoRA-GA   | `lora_variant="loraga"`                                                |
| BaLoRA-GA | `lora_variant="loraga"`, `project_every=1`                            |
| RefLoRA   | `lora_variant="reflora"`                                               |
| LoRA-RITE | `optimizer="lora_rite"`                                                |

`A_scaling` and `learning_rate` are the two swept hyperparameters.

---

## Reproducing the paper’s figures and tables

Sweeps are grouped by `model_dataset`; the mapping between paper items and the
retained sweeps is:

| Paper item | What it shows | Reproduce with |
|---|---|---|
| **Fig. 3** | Synthetic LoRA vs. BaLoRA loss (1- and 2-layer linear nets) | `synthetic/synthetic_loss.py` (set `shapes` for depth) |
| **Fig. 4 / Fig. 8** | Hyperparameter-sensitivity heatmaps, Llama-3.2-3B / Wikitext | `plotting/plot_sweeps.py` over `llama_wikitext/` sweeps 23 (LoRA), 22 (BaLoRA), 17 (OLoRA), 18 (LoRA-GA), 21 (DoRA) |
| **Fig. 7** | Test loss vs. runtime, Llama / Wikitext | `plotting/runtime_loss_plot.py` (`file_name="llama_wikitext"`) → sweeps 23, 22, 17, 18, 21, 23_lorarite, 23_reflora |
| **Fig. 9** | Test loss vs. runtime, Llama / GSM8K | `plotting/runtime_loss_plot.py` (`file_name="llama_gsm8k"`) → sweeps 25, 26, 32, 29, 31 |
| **Fig. 10** | Eval loss, Qwen-2.5-3B / Alpaca | `plotting/plot_sweeps.py` over `best_configs/` (Qwen, per-method best HP) |
| **Table 1** | Qwen, 5 datasets (Alpaca, CodeFeedback, OpenHermes, OpenOrca, WizardLM), r=8 | `best_configs/` single runs + `qwen_openhermes/sweep_94` (DoRA) |
| **Table 3** | Llama / GSM8K | `llama_gsm8k/` sweeps 25 (LoRA), 26 (BaLoRA), 31 (DoRA), 32 (OLoRA), 29 (LoRA-GA) |
| **Table 4** | Llama & Qwen / Wikitext | Llama: 23, 22, 21, 17, 18 · Qwen: `qwen_wikitext/` 60, 61, 64, 63, 65 |
| **Table 5** | Qwen / MetaMathQA | `qwen_metamath/` sweeps 70, 71, 74, 73, 75 + `best_configs/` |
| **Table 6** | Peak GPU memory | `experiments/profile_memory.py` |

**Not covered by this repo:** Fig. 5, Fig. 11, Table 7, and Table 8 (rank
ablations of Qwen-2.5-3B on DeepMind Mathematics and arXiv).

**Illustrative figures with no generating script:** Fig. 1 (method diagram), Fig. 2
(3D manifold intuition), Fig. 6 (condition-number histograms).

### Plotting workflow

The plotting scripts read fine-tuning results from `experiments/results/`:

- `plot_sweeps.py` expects per-run validation losses at
  `experiments/results/sweep_N/losses_lr_<i>_scaling_<j>.json` and is driven by a
  JSON plot config (see [`plotting/plot_configs/qwen_metamath.json`](experiments/plotting/plot_configs/qwen_metamath.json)):

  ```bash
  cd experiments/plotting
  python plot_sweeps.py --config plot_configs/qwen_metamath.json
  ```

- `runtime_loss_plot.py` reads TensorBoard event files, one folder per sweep,
  from `experiments/results/tensorboard/sweep_N/` (override with
  `$BALORA_EVENTS_DIR`). Select the panel with the `file_name` variable at the top.

Both write PDFs into a local `figures/` directory. `best_configs/` also contains
the helper `create_checkpoint_config.py`, which picks the best `(lr, scaling)` from
a reference sweep and emits a single-run “best config.”

---

## Synthetic experiments (Fig. 3)

```bash
cd synthetic
python synthetic_loss.py          # writes figures/synthetic_loss_adam_True.pdf
```

Edit the knobs at the top of `synthetic_loss.py`: `shapes` (`[(n, m)]` for the
one-layer panel, `[(n, m), (m, m)]` for two layers), `right_scaling` (init scaling
α/r), and `adam` (Adam vs. gradient descent).

---

## Citation

If you use this code, please cite the paper *Balanced LoRA: Removing Parameter
Invariance to Accelerate Convergence*.

```bibtex
@misc{castin2026balancedloraremovingparameter,
      title={Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence}, 
      author={Valérie Castin and Kimia Nadjahi and Pierre Ablin and Gabriel Peyré},
      year={2026},
      eprint={2605.31484},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.31484}, 
}
```

## License

See [`LICENSE`](LICENSE).
