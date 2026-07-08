#!/bin/bash
#SBATCH --job-name=QwenWikitextLoRAGA
#SBATCH --output=outputs/sweep_65/out_%A_%a.txt
#SBATCH --error=outputs/sweep_65/err_%A_%a.txt
#SBATCH --array=0-8
#SBATCH -A fgu@a100
#SBATCH -C a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

cd ${SLURM_SUBMIT_DIR}

module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

i=$((SLURM_ARRAY_TASK_ID / 3))
j=$((SLURM_ARRAY_TASK_ID % 3))

CONFIG_FILE="experiments/configs/qwen_wikitext/sweep_65/config/config_lr_${i}_scale_${j}.json"

echo "Running job $SLURM_ARRAY_TASK_ID: i=$i, j=$j, config=$CONFIG_FILE"

srun python train.py --config "$CONFIG_FILE"