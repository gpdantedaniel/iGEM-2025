#!/bin/bash
#SBATCH --account=def-jch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dante.garciaperez@mail.mcgill.ca

source venv/bin/activate  

python train-SpySeer-ESM2-650M-MSCRAMM-100K-TAKE2.py