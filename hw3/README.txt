### Usage ###
#1. training
#############
./train.sh $1 $2
$1: directory path contains (all_label.p, all_unlabel.p, test.p)
$2: output_model (if $2 is "model", the output model contains "model.json", "model_weights.h5")

#2. test
#############
./test.sh $1 $2 $3
$1: directory path contains (all_label.p, all_unlabel.p, test.p)
$2: input_model (directly use the output_model name, for example "model", this will load "model.json",
                 "model_weights.h5")
$3: prediction.csv

