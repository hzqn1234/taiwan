#!/bin/sh

#SBATCH -o gpu-job-run.output
#SBATCH -p HPCAIq
#SBATCH --gpus-per-node=0
#SBATCH -w node14
#SBATCH -c 16

export PYTHONUNBUFFERED=1

# python -u S1_data_prep.py

# python -u S2_FE.py

# python -u S4a_feature_sample.py

# python -u S4a_nn_series.py

# python -u S5_LGB_main_00.py

python -u S5_LGB_main_01.py

