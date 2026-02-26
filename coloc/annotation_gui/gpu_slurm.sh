#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --mem=50g
#SBATCH -t 00-06:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

#module add matlab/2019a
module add anaconda
module add gcc

SAMPLES="TEST1"
STAGE="count"

for sample in $SAMPLES
do
  matlab -nodesktop -nodisplay -nosplash -r "NM_config('$STAGE','$sample',true);"
done
