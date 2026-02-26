#!/bin/bash

SAMPLES="NF1R3M1R"	#or specify multiple samples like "TEST1 TEST2..."
STAGE="stitch"

for sample in $SAMPLES
do 
  matlab -nodesktop -nodisplay -nosplash -r "try NM_config('$STAGE','$sample',true); catch ME; fprintf('Error running sample  %s\n','$sample'); disp(ME.message); disp(ME.stack.line); end; exit;"
done 

echo "Completed All Samples"
