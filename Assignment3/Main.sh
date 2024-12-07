#!/bin/bash
#SBATCH -J 10-6781
#SBATCH --mem=100GB
#SBATCH --gpus=1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --partition=phys
#SBATCH -o COMP6781-A3-10epoch-%j.out

module load anaconda
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/COMP6781env
python3 Assignment3.py
