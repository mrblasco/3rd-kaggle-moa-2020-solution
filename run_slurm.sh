#!/bin/bash
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:1	      # Access to GPU
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-user=ablasco@fas.harvard.edu # Where to send mail
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
module load cuda/9.0-fasrc02
source activate my_env
#conda install pytorch pytorch-tabnet pandas scikit-learn numpy
#conda install -c trent-b iterative-stratification

# training -------------------------------------------------------
time python -W ignore -u daishu_solution.py #--data-dir=./data/expanded_targets/ --model-dir=./experiments/expanded_targets/
#time python -W ignore -u dnn-train.py
#time python -W ignore -u tabnet-train.py

