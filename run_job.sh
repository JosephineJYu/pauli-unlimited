#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH --output=submit.out
#SBATCH --error=submit.err
#SBATCH -p hns
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -n 31
#SBATCH --mem-per-cpu=4G

# email
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jjy@stanford.edu

NOWDIR=$(pwd)

ml python/3.6.1
ml py-scipy/1.4.1_py36
ml py-numba/0.35.0_py36
ml py-numpy/1.18.1_py36
pwd	
echo "running!"
python3 ex_calc_pair_susc.py 
