import argparse
import sys
from pathlib import Path
import os
import random
import pandas as pd


from utils_transfer.cal_metrics import cal_metrics, cal_metrics_non_normal
from utils_transfer.general import load_true_prediction, load_model_prediction, fillna, iou_conf_constraint

from utils_transfer.save import save_mapping, save_transfer_log, save_csv


ROOT_DIR = "/home/jovyan/Quan/Data-Competition"


true_train_path = os.path.join(ROOT_DIR, "dataset/labels/train")
true_val_path = os.path.join(ROOT_DIR, "dataset/labels/val")

model_train_path = os.path.join(ROOT_DIR, "prediction/transfer_train")
model_val_path = os.path.join(ROOT_DIR, "prediction/transfer_val")


def run(dir,
       iou_upper_thresh,
       conf_upper_thresh,
       conf_lower_thresh,
       iou_lower_thresh):
    
    
    # indentify dest folder
    destination_path = os.path.join(ROOT_DIR, dir)
    
    os.mkdir(destination_path)
    
    
    # load val true and predicted results 
    val_true_prediction = load_true_prediction(true_val_path)
    val_model_prediction = load_model_prediction(model_val_path)

    rest = []

    for epoch_pred in val_model_prediction:
        temp_rest = [txt for txt in val_true_prediction if txt not in epoch_pred]
        rest.append(temp_rest)

    for i in range(100):
        for missing_img in rest[i]:
            temp = []
            for _ in range(len(val_true_prediction.get(missing_img))):
                temp.append([5, 0, 0, 0, 0, 1])
            val_model_prediction[i][missing_img] = temp
            
            
    val_model_prediction = fillna(val_true_prediction, val_model_prediction)
    
    
    # load train true and predicted results
    true_prediction = load_true_prediction(true_train_path)
    model_prediction = load_model_prediction(model_train_path)

    rest = []

    for epoch_pred in model_prediction:
        temp_rest = [txt for txt in true_prediction if txt not in epoch_pred]
        rest.append(temp_rest)

    for i in range(100):
        for missing_img in rest[i]:
            temp = []
            for _ in range(len(true_prediction.get(missing_img))):
                temp.append([5, 0, 0, 0, 0, 1])
            model_prediction[i][missing_img] = temp

    model_prediction = fillna(true_prediction, model_prediction)
    
    print("done loading and filling")
    
    
    # calculate confidence, correctness and iou of each instance
    pred_pos, img_name, final_conf, final_cor, final_iou = cal_metrics(true_prediction, model_prediction)    
    val_pred_pos, val_img_name, val_final_conf, val_final_cor, val_final_iou = cal_metrics(val_true_prediction, val_model_prediction)  
    
    # save csv file
    both_pred_pos = pred_pos + val_pred_pos
    both_img_name = img_name + val_img_name
    both_final_conf = list(final_conf) + list(val_final_conf)
    both_final_cor = list(final_cor) + list(val_final_cor)
    both_final_iou = list(final_iou) + list(val_final_iou)
    
    csv_path = os.path.join(destination_path, "save_results")
    save_csv(both_img_name, both_pred_pos ,both_final_conf, both_final_cor, both_final_iou, csv_path)
    
    print("done save csv")
    
    # save training dynamic map
    table = pd.read_csv(csv_path)
    save_mapping(table, destination_path, "general_map.png")
    
    csv_path_val = os.path.join(destination_path, "save_results_val")
    save_csv(val_img_name,val_pred_pos, val_final_conf, val_final_cor, val_final_iou, csv_path_val)
    
    # save training dynamic map
    table = pd.read_csv(csv_path_val)
    save_mapping(table, destination_path, "val_map.png")
    
    print("done save map")


    # calculate confidence, correctness and iou of each image
    train_file_list, train_avg_conf, train_avg_cor, train_avg_iou = cal_metrics_non_normal(true_prediction, model_prediction)    
    val_file_list, val_avg_conf, val_avg_cor, val_avg_iou = cal_metrics_non_normal(val_true_prediction, val_model_prediction)    
    
    #filter images by iou and confidence constraints 
    train_high_iou_conf = iou_conf_constraint(train_file_list, 
                                              train_avg_conf, 
                                              train_avg_iou, 
                                              conf_upper_thresh, 
                                              iou_upper_thresh)
    
    val_low_iou_conf = iou_conf_constraint(val_file_list, 
                                           val_avg_conf, 
                                           val_avg_iou, 
                                           conf_lower_thresh, 
                                           iou_lower_thresh, 
                                            higher=False)
    
    
    # randomly pick images to exchange 
    rd_pos = random.sample(range(len(train_high_iou_conf)), len(val_low_iou_conf))
    train2val = [img for idx, img in enumerate(train_high_iou_conf) if idx in rd_pos]
    
    
    # save transfer logs
    save_transfer_log(train2val, destination_path)
    save_transfer_log(val_low_iou_conf, destination_path, False)
    
    print("save transfer log")
    
    
def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--dir', type=str, help='save results to dir', required=True)
    
    args = args.parse_args()
    
    args.iou_upper_thresh = 0.8
    args.conf_upper_thresh = 0.8
    
    args.iou_lower_thresh = 0.4
    args.conf_lower_thresh = 0.4

    return args

def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    main(parser())