#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --time=167:59:59
#SBATCH --partition=extended-96core
#SBATCH --output=array_test.%N_%a.log
#SBATCH --job-name=332
#SBATCH --mail-user=
#SBATCH --mail-type=ALL

module load slurm
module load  python39
module use /gpfs/projects/FrenkelGroup/shimogawa/modulefiles/FEFF
module load 10.0.0

#module load intel/oneAPI/2021.4

#module load  mpi/2021.4.0

##module load gnu-parallel

python run_FEFF.py -w
wait
python run_FEFF.py -r
