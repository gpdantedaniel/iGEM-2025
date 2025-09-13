#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

HF_HUB_OFFLINE="1"

source venv/bin/activate  

python train-SpySmith-650M.py