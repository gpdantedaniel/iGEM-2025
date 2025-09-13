#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

source venv/bin/activate  

python train-SpySeer-650M-1V-MSCRAMM-100K.py