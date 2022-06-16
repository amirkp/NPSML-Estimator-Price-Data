#!/bin/bash
#SBATCH --job-name=ak68
#SBATCH --partition=interactive
#SBATCH --ntasks=2
#SBATCH --ntasks-per-core=1
#SBATCH --mem-per-cpu=1000MB
#SBATCH --time=00:10:00
#SBATCH --output=test.log
julia MC-LN.jl
