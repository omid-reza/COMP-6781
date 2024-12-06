#!/bin/bash
#SBATCH -J 100E-6781
#SBATCH --mem=150GB
#SBATCH --gpus=2
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=all

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/COMP6781env
python3 Assignment3.py
