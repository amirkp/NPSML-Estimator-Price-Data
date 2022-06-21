#!/bin/bash
#SBATCH --job-name=ak68
#SBATCH --partition=scavenge
#SBATCH --ntasks=100
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=04:00:00
#SBATCH --output=out_%A_n_200_h.log
julia MC-LN.jl
