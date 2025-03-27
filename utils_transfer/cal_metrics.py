import numpy as np


from utils.metrics import box_iou
from utils_transfer.general import true_pred_bboxes, model_pred_bboxes


def optimal_order(mat):

    current_iou = np.zeros((len(mat), len(mat[0])))
    table= []

    for i in range(len(mat)):
        table.append([])
        for _ in range(len(mat[0])):
            table[i].append([])

    for i in range(len(mat[0])):
        current_iou[-1][i] = mat[-1][i]
        table[-1][i].append([i])

    for i in range(len(mat) - 2, -1, -1):
        for j in range(len(mat[0])):
            for idx, paths in enumerate(table[i+1]):            
                for path in paths:
                    if j not in path:

                        if mat[i][j] + current_iou[i+1][idx] > current_iou[i][j]:
                            current_iou[i][j] = mat[i][j] + current_iou[i+1][idx]
                            table[i][j] = [[j]+path]
                        elif mat[i][j] + current_iou[i+1][idx] == current_iou[i][j]:
                            table[i][j].append([j]+path)

    pos_max = np.argmax(np.array(current_iou[0]))

    return table[0][pos_max]
    

def cal_img_metrics(true_pred, epoch_pred):
    
    true_bboxes = true_pred_bboxes(true_pred)
    pred_bboxes = model_pred_bboxes(epoch_pred)

    iou_mat = box_iou(true_bboxes, pred_bboxes)

    orders = optimal_order(iou_mat)
    
    correctness_map = np.zeros((len(true_bboxes),len(pred_bboxes)))
    for true_pos in range(len(true_bboxes)):
        for pred_pos in range(len(pred_bboxes)):
            if true_pred[true_pos][0] == epoch_pred[pred_pos][0]:
                correctness_map[true_pos][pred_pos] = 1
                
    if len(orders) > 1:
        max_correctness = 0
        best_order = []
        for order in orders : 
            order_correctness = 0
            for true_pos, pred_pos in enumerate(order):
                order_correctness += correctness_map[true_pos][pred_pos]
            if order_correctness >= max_correctness:
                best_order = order
    else: 
        best_order = orders[0]

    iou = [iou_mat[i][value] for i, value in enumerate(best_order)]
    correctness_map = correctness_map.tolist()
    cor = [correctness_map[i][value] for i, value in enumerate(best_order)]
    conf = [epoch_pred[value][-1] if cor[i] == 1 else (1-epoch_pred[value][-1]) / 4 for i, value in enumerate(best_order)]

    return cor, conf, iou

def cal_epoch_metrics_non_normal(true_prediction, epoch_prediction):
    iou = []
    conf = []
    correct = []
    
    for key, values in true_prediction.items():
        img_cor, img_conf ,img_iou = cal_img_metrics(values, epoch_prediction[key])
        
        iou.append(np.sum(img_iou) / len(values))
        conf.append(np.sum(img_conf) / len(values))
        correct.append(np.sum(img_cor) / len(values))
    
    return correct, conf, iou


def cal_epoch_metrics(true_prediction, epoch_prediction):
    iou = []
    conf = []
    correct = []

    
    for key, values in true_prediction.items():
        img_cor, img_conf ,img_iou = cal_img_metrics(values, epoch_prediction[key])
        
        iou += img_iou
        conf += img_conf
        correct += img_cor
            
    return correct, conf, iou


def cal_metrics_non_normal(true_prediction, model_prediction, iter_num):
    print("cal not normal metrics")
    
    img_name = []
    
   
    for key, values in true_prediction.items():
        img_name.append(key)
    
      
    total_conf = np.zeros(len(img_name))
    total_cor = np.zeros(len(img_name))
    total_iou = np.zeros(len(img_name))
    
    
    for idx, epoch_prediction in enumerate(model_prediction):
    # return a dict of results for each epoch
        epoch_correctness, epoch_conf, epoch_iou = cal_epoch_metrics_non_normal(true_prediction, epoch_prediction)
        
        total_conf = np.add(total_conf, np.array(epoch_conf))
        total_cor = np.add(total_cor, np.array(epoch_correctness))
        total_iou = np.add(total_iou, np.array(epoch_iou))
                
    
    return img_name, total_conf / iter_num, total_cor / iter_num, total_iou / iter_num 



def cal_metrics(true_prediction, model_prediction, iter_num):      
    print("cal normal metrics")
    pred_pos = []
    img_name = []
    

    for key, values in true_prediction.items():
        for i in range(len(values)):
            pred_pos.append(i)
            img_name.append(key)
    
    total_conf = np.zeros(len(img_name))
    total_cor = np.zeros(len(img_name))
    total_iou = np.zeros(len(img_name))
    
    for idx, epoch_prediction in enumerate(model_prediction):
        # return a dict of results for each epoch
        epoch_correctness, epoch_conf, epoch_iou = cal_epoch_metrics(true_prediction, epoch_prediction)
        print("done epoch")
        

        total_conf = np.add(total_conf, np.array(epoch_conf))
        total_cor = np.add(total_cor, np.array(epoch_correctness))
        total_iou = np.add(total_iou, np.array(epoch_iou))
                

    return pred_pos, img_name, total_conf / iter_num, total_cor / iter_num, total_iou / iter_num
    