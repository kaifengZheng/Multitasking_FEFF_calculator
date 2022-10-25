#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=167:59:59
#SBATCH --partition=extended-40core
#SBATCH --output=array_test.%N_%a.log
#SBATCH --job-name=kaif
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

module load slurm
module load  python39
module use module
module load 10.0.0

#module load intel/oneAPI/2021.4

#module load  mpi/2021.4.0

##module load gnu-parallel

python run_FEFF.py -r	 
