===========================================================
# reorganize the flat ISIC training images 
# into class subfolders required by datasets.ImageFolder

# E.g.:
# ISIC_2019_Training_Organized/
#    AK/
#        image1.jpg
#        ...
#    BCC/
#        image2.jpg
#        ...
# ===========================================================
#!/bin/bash
#SBATCH --job-name=organize_isic
#SBATCH --partition=standard
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=shakeri_ds6050
#SBATCH --output=organize_isic_%j.log

cd /home/vrb9e/DS6050_Deep-Learning/group-project/code

module load anaconda
source activate dl-course

echo "Start: $(date)"
python organize_isic.py
echo "End: $(date)"
