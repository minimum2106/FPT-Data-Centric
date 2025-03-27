import numpy as np
import torch 
import os
from io import StringIO
import copy

from utils.general import xywhn2xyxy


def load_true_prediction(path):
    true_prediction = {}

    for filename in os.listdir(path):
        if not filename == ".ipynb_checkpoints":
            img_path = os.path.join(path, filename)
            temp = []
            with open(img_path) as f : 
                lines = f.readlines()
                for line in lines : 
                    c = StringIO(line)
                    num_line = np.loadtxt(c)
                    temp.append(num_line)

            true_prediction[f"{filename}"] = temp

    true_prediction = {key : value for key, value in true_prediction.items() if value}
    
    return true_prediction


def load_model_prediction(path, detect_freq):
    model_prediction = []

    for i in range(0, 100, detect_freq):
        epoch_prediction = {}
        
        epoch_path = os.path.join(path, f"epoch_{i}/labels")
        
        for filename in os.listdir(epoch_path):
            if not filename == ".ipynb_checkpoints":
                img_path = os.path.join(epoch_path, filename)
                temp = []
                with open(img_path) as f : 
                    lines = f.readlines()
                    for line in lines : 
                        c = StringIO(line)
                        num_line = np.loadtxt(c)
                        temp.append(num_line)
                epoch_prediction[filename] = temp
        model_prediction.append(epoch_prediction)
    
    return model_prediction


def fillna(true_prediction, model_prediction):
    non_null_model_prediction = copy.deepcopy(model_prediction)
    
    for idx, epoch_prediction in enumerate(non_null_model_prediction):
        for key, value in epoch_prediction.items():
            if len(epoch_prediction[key]) < len(true_prediction[key]):
                for _ in range(len(true_prediction[key]) - len(epoch_prediction[key])):
                    epoch_prediction[key].append([5, 0, 0, 0, 0, 1])
    
    return non_null_model_prediction
    
def true_pred_bboxes(true_prediction):

    pred_tensor = torch.Tensor(true_prediction)
    bboxes = pred_tensor[:, 1:]
    bboxes = xywhn2xyxy(bboxes)
    
    return bboxes

def model_pred_bboxes(model_prediction):
 
    pred_tensor = torch.Tensor(model_prediction)
    bboxes = xywhn2xyxy(pred_tensor[:, 1:-1])

    return bboxes

def iou_conf_constraint_w_pos(file_list, 
                                pos_list,
                                avg_conf, 
                                avg_iou, 
                                conf_thresh, 
                                iou_thresh,
                               higher=True):

    if higher : 
        qualified_conf_pos = [idx for idx, conf in enumerate(avg_conf) if conf > conf_thresh]
        qualified_iou_pos = [idx for idx, iou in enumerate(avg_iou) if iou > iou_thresh]
       
    else: 
        qualified_conf_pos = [idx for idx, conf in enumerate(avg_conf) if conf < conf_thresh]
        qualified_iou_pos = [idx for idx, iou in enumerate(avg_iou) if iou < iou_thresh]
        
    satis_both = [pos for pos in qualified_conf_pos if pos in qualified_iou_pos]
    
    satis_img = [img for idx, img in enumerate(file_list) if idx in satis_both]
    satis_pos = [pos for idx, pos in enumerate(pos_list) if idx in satis_both]
    
    return satis_img, satis_pos


def iou_conf_constraint(file_list, 
                        avg_conf, 
                        avg_iou, 
                        conf_thresh, 
                        iou_thresh,
                       higher=True):
    
    if higher : 
        qualified_conf_pos = [idx for idx, conf in enumerate(avg_conf) if conf > conf_thresh]
        qualified_iou_pos = [idx for idx, iou in enumerate(avg_iou) if iou > iou_thresh]
       
    else: 
        qualified_conf_pos = [idx for idx, conf in enumerate(avg_conf) if conf < conf_thresh]
        qualified_iou_pos = [idx for idx, iou in enumerate(avg_iou) if iou < iou_thresh]
        
    satis_both = [pos for pos in qualified_conf_pos if pos in qualified_iou_pos]
    
    satis_img = [img for idx, img in enumerate(file_list) if idx in satis_both]
    
    return satis_img

