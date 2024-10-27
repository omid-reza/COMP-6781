#!/bin/bash
#SBATCH -J HW2_50k
#SBATCH --mem=200GB
#SBATCH -p phys
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omid.orh@gmail.com
#SBATCH --gpus=1

module load anaconda/default
eval "$(conda shell.bash hook)"
conda activate /home/o_heidar/yoloxenv
python hw2.py