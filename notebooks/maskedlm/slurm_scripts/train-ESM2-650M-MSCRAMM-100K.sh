#!/bin/bash
#SBATCH --account=def-jch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=v100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dante.garciaperez@mail.mcgill.ca

cd dante_workspace
source venv/bin/activate

python train-SpySeer-ESM2-150M-MSCRAMM-100K.py