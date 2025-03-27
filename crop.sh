#!/bin/bash 

# Quan
# 12/11/2021 
# Script to augment images between train_dataset and val_dataset


for i in {0..9}; do
    
    # loop the whole process in 10 times
    python train.py --batch-size 32 --name crop_purpose;
    
    # retrieve new results with new train weigths
    for j in {0..99..5}; do
        python3 detect.py --device 0 \
          --save-txt \
          --save-conf \
          --half \
          --weights results/train/crop_purpose/weights/epoch_$j.pt \
          --source dataset/images/val \
          --dir prediction/crop_val/epoch_$j; 

          
        python3 detect.py --device 0 \
          --save-txt \
          --save-conf \
          --half \
          --weights results/train/crop_purpose/weights/epoch_$j.pt \
          --source dataset/images/train \
          --dir prediction/crop_train/epoch_$j; 
          
    done
    
    # cal all metrics and save this iter info 
    python3 crop.py --dir crop_results/iter_$i \
                    --var_num 5 \
                    --detect_freq 5;
                
    

    cd prediction;
    rm -r crop_val;
    rm -r crop_train;
    cd .. ;
    
    cd results;
    cd train;
    rm -r crop_purpose/weights/epoch*;
    mv crop_purpose usth_epoch_$i;
    cd ../..;
    
done 

exit
