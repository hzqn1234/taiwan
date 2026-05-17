#!/bin/sh

#SBATCH -o gpu-job-run.output
#SBATCH -p HPCAIq
#SBATCH --gpus-per-node=0
#SBATCH -w node14
#SBATCH -c 16

export PYTHONUNBUFFERED=1

python -u S1_data_prep.py

python -u S2_FE.py


