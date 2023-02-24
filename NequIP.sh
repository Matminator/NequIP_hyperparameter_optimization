#!/bin/bash -le
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:RTX3090:1

source $HOME/thesis/thesis_venv/bin/activate

nequip-train TRAIN
nequip-deploy DEPLOY
python MODEL_EVALUATOR


