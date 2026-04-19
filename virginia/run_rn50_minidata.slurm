#!/bin/bash
#SBATCH --job-name=resnet50_30ep_minidata
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=shakeri_ds6050
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=resnet50_30ep_%j.log

# --- Paths (update when using full/mini(on local)) --- #
DATA_DIR="/home/vrb9e/DS6050_Deep-Learning/group-project/data/ISIC_2019_Training_Organized"
SCRIPT_DIR="/home/vrb9e/DS6050_Deep-Learning/group-project/code"

cd $SCRIPT_DIR

# --- Environment --- #
# module load anaconda # not needed on rivanna
eval "$(conda shell.bash hook)"
conda activate dl-course

# --- W&B login (only needed first time; comment out after) --- #
# wandb login YOUR_API_KEY

echo "Start: $(date)"

python rn50_base_va.py \
    --data_dir $DATA_DIR \
    --num_epochs 30 \
    --batch_size 32 \
    --num_workers 2 \
    --lr 0.001 \
    --wandb

echo "End: $(date)"
