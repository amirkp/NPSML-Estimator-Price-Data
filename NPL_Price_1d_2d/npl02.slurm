#!/bin/bash
#SBATCH --job-name=ak68
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=10:00:00
#SBATCH --array=1
#SBATCH --output=out_%A_%a.est

srun julia -p $(( 2 * $SLURM_CPUS_PER_TASK )) npl_02.jl
