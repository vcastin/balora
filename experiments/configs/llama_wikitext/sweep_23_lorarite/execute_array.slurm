#!/bin/bash
#SBATCH --job-name=LoraRITE
#SBATCH --output=outputs/sweep_23_lorarite/out_%A_%a.txt
#SBATCH --error=outputs/sweep_23_lorarite/err_%A_%a.txt
#SBATCH --array=0-49
#SBATCH -A wbg@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

cd ${SLURM_SUBMIT_DIR}

module purge
module load arch/a100
module load pytorch-gpu/py3/2.6.0

i=$((SLURM_ARRAY_TASK_ID / 5))
j=$((SLURM_ARRAY_TASK_ID % 5))

CONFIG_FILE="experiments/configs/llama_wikitext/sweep_23_lorarite/config/config_lr_${i}_scale_${j}.json"

echo "Running job $SLURM_ARRAY_TASK_ID: i=$i, j=$j, config=$CONFIG_FILE"

srun python train.py --config "$CONFIG_FILE"