#!/bin/bash
#SBATCH --account=def-jch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dante.garciaperez@mail.mcgill.ca

cd dante_workspace
source venv/bin/activate

python train-SpySeer-ESM1V-150M-MSCRAMM-100K.py