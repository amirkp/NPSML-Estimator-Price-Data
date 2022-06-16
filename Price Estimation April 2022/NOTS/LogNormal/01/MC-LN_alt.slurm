#!/bin/bash
#SBATCH --job-name=ak68
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00
#SBATCH --array=1
#SBATCH --output=test.log

srun julia -p $(( 2 * $SLURM_CPUS_PER_TASK )) alt_MC-LN.jl
