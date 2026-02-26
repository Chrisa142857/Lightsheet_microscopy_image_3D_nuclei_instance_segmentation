#! /bin/bash

RES="121"
N=4

if [ "$RES" == "075" ];
then
   matlab -nodesktop -nodisplay -nosplash -r "res='$RES'; file_delimiter='condensed'; run preprocess_3dunet(res,file_delimiter).m; exit;"
   python train_isensee2017_2.py

elif [ "$RES" == "121" ];
then
   matlab -nodesktop -nodisplay -nosplash -r "res='$RES'; file_delimiter='final'; n=$N; run preprocess_3dunet(res,file_delimiter,n).m; exit;"
   python train_isensee2017_3.py
   python predict_validation.py
   bash bash_evaluate.sh
fi
bash bash_evaluate.sh
