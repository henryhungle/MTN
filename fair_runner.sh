#!/bin/bash
## SLURM scripts have a specific format. 

## job name
#SBATCH --job-name=mtn
#SBATCH --output=/checkpoint/%u/logs/mtn/%x-%j.out
#SBATCH --error=/checkpoint/%u/logs/mtn/%x-%j.err

## partition name
#SBATCH --partition=learnfair
## number of nodes
#SBATCH --nodes=1
#SBATCH --time=4320
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
## number of tasks per node
#SBATCH --ntasks-per-node=1

srun --label run.sh 1 resnet-f1 resnet-f1 100 1000 0
