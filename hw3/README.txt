### Usage ###
#1. train
#############
./train.sh $1 $2 $3
$1: directory path contains (all_label.p, all_unlabel.p, test,p)
$2: output_model for encoder (if $2 is "model", the output_model will be "model.h5")
$3: output_model for dnn (if $3 is "model", the output_model contains "model.json" and "model_weights.h5")

#2. test
#############
./test.sh $1 $2 $3 $4
$1: directory path contains (all_label.p, all_unlabel.p, test,p)
$2: input_model for encoder (directly use the output_model name, for example "model", this will load "model.h5")
$3: input_model for dnn (directly use the output_model name, for example "model", this will load "model.json" and
                         "model_weights.h5")
$4: prediction.csv

