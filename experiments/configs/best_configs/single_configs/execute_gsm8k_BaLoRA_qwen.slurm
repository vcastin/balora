#!/bin/bash
#SBATCH --job-name=gsm8k_BaLoRA
#SBATCH --output=outputs/balora_gsm8k_qwen/out_%A_%a.txt
#SBATCH --error=outputs/balora_gsm8k_qwen/err_%A_%a.txt
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --array=0-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0

CONFIG_DIR="$WORK/balanced-lora/experiments/configs/best_configs/single_configs"

FILES=("$CONFIG_DIR"/qwen_gsm8k_BaLoRA*.json)

CONFIG_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "------------------------------------------------------------"
echo "JOB ID: ${SLURM_ARRAY_JOB_ID}"
echo "TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "CONFIG FILE: ${CONFIG_FILE}"
echo "DATE: $(date)"
echo "------------------------------------------------------------"

srun python train.py --config "$CONFIG_FILE"