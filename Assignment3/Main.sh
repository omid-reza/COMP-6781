#!/bin/bash
#SBATCH -J COMP6781
#SBATCH --mem=200GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=all
#SBATCH --chdir=../../aldi/
#SBATCH -o ../jobs/outputs/CFC-ALDI-FPN-%j.out

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/COMP6781env
python3 Assignment3.py