#!/bin/sh

#SBATCH -o gpu-job-run.output
#SBATCH -p PA100q
#SBATCH --gpus-per-node=0
#SBATCH -w node03
#SBATCH -c 16

export PYTHONUNBUFFERED=1

# python -u S1_data_prep.py

# python -u S2_FE.py

# python -u S4a_feature_sample.py

# python -u S4a_nn_series.py

# python -u S5_LGB_main_00.py

# python -u S5_LGB_main_01.py

# python -u S5_LGB_main_02.py

# python -u S5_LGB_main_03.py

# python -u S5_LGB_main_04.py

# python -u S6_0_NN_PreProcess_0.py

# python -u S6_0_NN_PreProcess_1.py 

# CUDA_VISIBLE_DEVICES=4 python -u S6_1_NN_main_train.py --do_train --remark 'tsf + gru, lr=1000e-7'

# CUDA_VISIBLE_DEVICES=4 python -u  S6_2_NN_main_test.py

# CUDA_VISIBLE_DEVICES=4 python -u  S6_1_NN_main_train.py --do_train --use_fe --epochs 15 --lr 11000e-7 --remark 'tsf (2 layer, 0.005 dropout, dim_ff=4) + gru (2-dir), CosineAnnealingLR lr=9000e-7, 32 hidden, fe dropout=0.005, DNN=2, output batchnorm * 1. output dropout=0.025, epochs=15'

# CUDA_VISIBLE_DEVICES=4 python -u  S6_2_NN_main_test.py --use_fe

# python -u S7_ensemble.py

python -u S8_check_submission.py