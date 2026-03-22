#!/bin/bash
#SBATCH --job-name=parallel_3models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=shakeri_ds6050
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=parallel_3models_%j.log

# --- Paths (update when using full/mini) --- #
DATA_DIR="/home/vrb9e/DS6050_Deep-Learning/group-project/data/ISIC_2019_Training_Organized"
SCRIPT_DIR="/home/vrb9e/DS6050_Deep-Learning/group-project/code"

cd $SCRIPT_DIR

# --- Environment --- #
# module load anaconda # not needed on rivanna
source activate dl-course  # my virtual env for DS 6050

# --- W&B login (only needed first time; comment out after) --- #
# wandb login YOUR_API_KEY

echo "Start: $(date)"

# num_workers=0 on MacM3 if running locally with minids
python runner_parallel.py \
    --data_dir $DATA_DIR \
    --num_epochs 15 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 0.001 \
    --wandb

echo "End: $(date)"
