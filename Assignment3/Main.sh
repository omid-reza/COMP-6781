#!/bin/bash
#SBATCH -J 50-6781
#SBATCH --mem=85GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=phys
#SBATCH -o COMP6781-A3-50epoch.out
#SBATCH -w virya2

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/COMP6781env
python3 Assignment3.py
