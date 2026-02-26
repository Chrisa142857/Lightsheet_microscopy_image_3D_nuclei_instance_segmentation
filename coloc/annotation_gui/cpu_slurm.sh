#!/bin/bash

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p steinlab
#SBATCH --mem=50g
#SBATCH -t 01-00:00:00

#module add matlab/2019a
module add anaconda
module add gcc

SAMPLES="TEST1"
STAGE="process"

for sample in $SAMPLES
do
  matlab -nodesktop -nodisplay -nosplash -singleCompThread -r "myCluster = parcluster; myCluster.NumWorkers = 12; parpool(myCluster); NM_config('$STAGE','$sample',true);"
done
