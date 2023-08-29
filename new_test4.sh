#!/bin/bash
 
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=300GB
#PBS -l jobfs=1GB
#PBS -q gpuvolta
#PBS -P nm03
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/x77+gdata/x77+gdata/nm03
#PBS -l wd
  
export JULIA_DEPOT_PATH=/g/data/x77/dj0263/.julia:$JULIA_ROOT/share/julia/site/
export JULIA_CUDA_USE_BINARYBUILDER="false"

cd /g/data/nm03/dj0263/mean_stress_test

/g/data/x77/dj0263/julia-1.9/julia new_test4.jl > $PBS_JOBID.log
