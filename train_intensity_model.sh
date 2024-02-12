#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --partition=aoraki_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=10:00:00

source ~/miniconda3/bin/activate
conda activate nhenv

python train_model.py -d conttime_1000_Seq_5_Dim_50_MaxLen_8_LSTMUnits -e 15 -lr 0.01 