#!/bin/bash
#SBATCH --job-name=ak68
#SBATCH --partition=commons
#SBATCH --ntasks=50
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=10:00:00
#SBATCH --output=out_%A_n_hvary_h.log
julia MC-LN04.jl
