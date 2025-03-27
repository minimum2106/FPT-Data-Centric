import argparse
import sys
from pathlib import Path
import os
import random
import pandas as pd
import cv2


from utils_transfer.cal_metrics import cal_metrics
from utils_transfer.general import load_true_prediction, load_model_prediction, fillna, iou_conf_constraint_w_pos

from utils_transfer.save import save_mapping, save_transfer_log, save_csv, save_crop_log, save_img, save_txt

from crop_class.FaceAugment import FaceAugment, Label


ROOT_DIR = "/home/jovyan/Quan/Data-Competition"

img_train_path =  os.path.join(ROOT_DIR, "dataset/images/train")
img_val_path =  os.path.join(ROOT_DIR, "dataset/images/val")

true_train_path = os.path.join(ROOT_DIR, "dataset/labels/train")
true_val_path = os.path.join(ROOT_DIR, "dataset/labels/val")

model_train_path = os.path.join(ROOT_DIR, "prediction/crop_train")
model_val_path = os.path.join(ROOT_DIR, "prediction/crop_val")


def run(dir,
       iou_upper_thresh,
       conf_upper_thresh,
       conf_lower_thresh,
       iou_lower_thresh,
       var_num,
        detect_freq):
    
    ITER_NUM = int(100 / detect_freq)
    
    
    # indentify dest folder
    destination_path = os.path.join(ROOT_DIR, dir)
    
    os.mkdir(destination_path)
    
    
    # load val true and predicted results 
    val_true_prediction = load_true_prediction(true_val_path)
    val_model_prediction = load_model_prediction(model_val_path, detect_freq)

    rest = []

    for epoch_pred in val_model_prediction:
        temp_rest = [txt for txt in val_true_prediction if txt not in epoch_pred]
        rest.append(temp_rest)

    for i in range(ITER_NUM):
        for missing_img in rest[i]:
            temp = []
            for _ in range(len(val_true_prediction.get(missing_img))):
                temp.append([5, 0, 0, 0, 0, 1])
            val_model_prediction[i][missing_img] = temp
            
            
    val_model_prediction = fillna(val_true_prediction, val_model_prediction)
    
    
    # load train true and predicted results
    true_prediction = load_true_prediction(true_train_path)
    model_prediction = load_model_prediction(model_train_path, detect_freq)

    rest = []

    for epoch_pred in model_prediction:
        temp_rest = [txt for txt in true_prediction if txt not in epoch_pred]
        rest.append(temp_rest)

    for i in range(ITER_NUM):
        for missing_img in rest[i]:
            temp = []
            for _ in range(len(true_prediction.get(missing_img))):
                temp.append([5, 0, 0, 0, 0, 1])
            model_prediction[i][missing_img] = temp

    model_prediction = fillna(true_prediction, model_prediction)
    
    print("done loading and filling")
    
    
    # calculate confidence, correctness and iou of each instance
    pred_pos, img_name, final_conf, final_cor, final_iou = cal_metrics(true_prediction, model_prediction, ITER_NUM)    
    val_pred_pos, val_img_name, val_final_conf, val_final_cor, val_final_iou = cal_metrics(val_true_prediction, val_model_prediction, ITER_NUM)  
    
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
    
    val_satis_img, val_satis_pos = iou_conf_constraint_w_pos(val_img_name, 
                                                             val_pred_pos,
                                                             val_final_conf, 
                                                             val_final_iou, 
                                                             conf_lower_thresh, 
                                                             iou_lower_thresh, 
                                                             higher=False)

    train_satis_img, train_satis_pos = iou_conf_constraint_w_pos(img_name,
                                                                 pred_pos,
                                                                final_conf, 
                                                                final_iou, 
                                                                conf_lower_thresh, 
                                                                iou_lower_thresh, 
                                                                higher=False)
    
    face_augment = FaceAugment()

    print("gen train")
    print(len(train_satis_img))
    
    for i in range(len(train_satis_img)):
        img_path = os.path.join(img_train_path, train_satis_img[i].split('.')[0] + '.jpg')
        label_path = os.path.join(true_train_path, train_satis_img[i])
        
        print(train_satis_img[i], train_satis_pos[i])
        
        label = Label()
        label.extract_from_file(label_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                
        for j in range(var_num):
            try:
                new_img, new_labels = face_augment.augment_face(img, label, train_satis_pos[i])
                new_labels.correct_bounding_box()

                if len(new_labels.labels) > 0:
                    new_img_filename = train_satis_img[i].split('.')[0] + f'_{j}' + '.jpg'
                    new_label_filename = train_satis_img[i].split('.')[0] + f'_{j}' + '.txt'

                    new_labels.save(os.path.join(true_train_path, new_label_filename))            
                    cv2.imwrite(os.path.join(img_train_path, new_img_filename),new_img)
                
            except:
                continue

            
            
    print("gen val")
    print(len(val_satis_img))
    for i in range(len(val_satis_img)):
        img_path = os.path.join(img_val_path, val_satis_img[i].split('.')[0] + '.jpg')
        label_path = os.path.join(true_val_path, val_satis_img[i])
        
        print(val_satis_img[i], val_satis_pos[i])

        label = Label()
        label.extract_from_file(label_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        
        for j in range(var_num):
            try:
                new_img, new_labels =face_augment.augment_face(img, label, val_satis_pos[i])
                new_labels.correct_bounding_box()

                if len(new_labels.labels) > 0:
                    new_img_filename = train_satis_img[i].split('.')[0] + f'_{j}' + '.jpg'
                    new_label_filename = train_satis_img[i].split('.')[0] + f'_{j}' + '.txt'

                    new_labels.save(os.path.join(true_train_path, new_label_filename))            
                    cv2.imwrite(os.path.join(img_train_path, new_img_filename),new_img)
                    
                    
            except:
                continue

    save_crop_log(train_satis_img, train_satis_pos, destination_path)
    save_crop_log(val_satis_img, val_satis_pos, destination_path, fr_train=False)
    
     
    
def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--dir', type=str, help='save results to dir', required=True)
    args.add_argument('--var_num', type=int, help='number of variation from 1 image', required=True)
    args.add_argument('--detect_freq' , type=int, help='run detect.py frequency', required=True)
    
    
    args = args.parse_args()
    
    args.iou_upper_thresh = 0.8
    args.conf_upper_thresh = 0.8
    
    args.iou_lower_thresh = 1
    args.conf_lower_thresh = 0.3    
    

    return args

def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    main(parser())