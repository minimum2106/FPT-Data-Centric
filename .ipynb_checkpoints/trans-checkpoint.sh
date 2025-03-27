#!/bin/bash 

# Quan
# 12/11/2021 
# Script to transfer images between train_dataset and val_dataset

cd ..;


for i in {0..9}; do
    
    # loop the whole process in 10 times
    python train.py --batch-size 32 --name transfer_purpose;
    
    # retrieve new results with new train weigths
    for j in {0..99}; do
        python3 detect.py --device 0 \
          --save-txt \
          --save-conf \
          --half \
          --weights results/train/transfer_purpose/weights/epoch_$j.pt \
          --source dataset/images/val \
          --dir prediction/transfer_val/epoch_$j; 

      
       python3 detect.py --device 0 \
          --save-txt \
          --save-conf \
          --half \
          --weights results/train/transfer_purpose/weights/epoch_$j.pt \
          --source dataset/images/train \
          --dir prediction/transfer_train/epoch_$j; 
          
    done
    
    
    # cal all metrics and save this iter info 
    python3 transfer.py --dir transfer_results/iter_$i;
    
    
    # transfer data 
    TRAIN_LABELS="transfer_results/iter_$i/train2val.txt"
    VAL_LABELS="transfer_results/iter_$i/val2train.txt"
    
    
    #TRANSFER DATA FR TRAIN TO VAL 
    TRAIN_FILES=$(cat $TRAIN_LABELS)
    VAL_FILES=$(cat $VAL_LABELS)
    
    for LINE in $TRAIN_FILES;
    do
            mv "dataset/labels/train/${LINE}" dataset/labels/val/;
            
            mv "dataset/images/train/${LINE%.txt}.jpg" dataset/images/val/;
    done
    
    # TRANSFER DATA FR VAL TO TRAIN
    for LINE in $VAL_FILES;
    do
            mv "dataset/labels/val/${LINE}" dataset/labels/train/;
            
            mv "dataset/images/val/${LINE%.txt}.jpg" dataset/images/train/;
    done
    

    cd prediction;
    rm -r transfer_val;
    rm -r transfer_train;
    cd .. ;
    
    cd results;
    cd train;
    rm -r transfer_purpose/weights/epoch*;
    mv transfer_purpose usth_epoch_$i;
    cd ../..;
    
done 



exit
