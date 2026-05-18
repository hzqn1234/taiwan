#!/bin/sh

#SBATCH -o gpu-job-run.output
#SBATCH -p PA100q
#SBATCH --gpus-per-node=1
#SBATCH -w node03
#SBATCH -c 16

export PYTHONUNBUFFERED=1

# python -u S1_data_prep.py

# python -u S2_FE.py

# python -u S4a_feature_sample.py

# python -u S4a_nn_series.py

# python -u S5_LGB_main_00.py

# python -u S5_LGB_main_01.py

# python -u S6_0_NN_PreProcess_0.py

# python -u S6_0_NN_PreProcess_1.py 

CUDA_VISIBLE_DEVICES=4 python -u S6_1_NN_main_train.py --do_train --remark 'tsf + gru, lr=1000e-7'

