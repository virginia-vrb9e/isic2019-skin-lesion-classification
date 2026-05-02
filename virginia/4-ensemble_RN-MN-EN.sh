#!/bin/bash
#SBATCH -A shakeri_ds6050
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --output=/home/vrb9e/DS6050_Deep-Learning/group-project/code/ensemble_RN-EN-MN_%j.out

eval "$(conda shell.bash hook)"
conda activate dl-course
cd /home/vrb9e/DS6050_Deep-Learning/group-project/code
python hard_soft_vote_ensemb_allmetrics_wandb.py
 