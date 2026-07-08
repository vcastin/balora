#!/bin/bash
#SBATCH --job-name=LoraSimple
#SBATCH --output=outputs/sweep_18/out_%A_%a.txt
#SBATCH --error=outputs/sweep_18/err_%A_%a.txt
#SBATCH --array=32-49
#SBATCH -A wbg@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --hint=nomultithread

cd ${SLURM_SUBMIT_DIR}

module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0

i=$((SLURM_ARRAY_TASK_ID / 5))
j=$((SLURM_ARRAY_TASK_ID % 5))

CONFIG_FILE="experiments/configs/llama_wikitext/sweep_18/config/config_lr_${i}_scale_${j}.json"

echo "Running job $SLURM_ARRAY_TASK_ID: i=$i, j=$j, config=$CONFIG_FILE"

srun python train.py --config "$CONFIG_FILE"
