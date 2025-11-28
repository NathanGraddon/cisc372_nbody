#!/bin/bash -l
#SBATCH --job-name=nbody1000
#SBATCH --partition=gpu-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output nbody1000-job_%j.out
#SBATCH --error  nbody1000-job_%j.err

srun ./nbody
