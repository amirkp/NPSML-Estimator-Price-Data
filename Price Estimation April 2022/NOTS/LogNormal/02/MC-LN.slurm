#!/bin/bash
#SBATCH --job-name=ak68
#SBATCH --partition=scavenge
#SBATCH --ntasks=100
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=800MB
#SBATCH --time=03:59:00
#SBATCH --output=out_1.log
julia MC-LN.jl
