import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import csv
import cv2



def save_mapping (data, folder, filename):
    plt.figure(figsize=(11,11))
    sns.scatterplot(x="iou", y="confidence", hue="correctness", data = data)
    plt.text(0.01, 0.1, "hard-to-learn", horizontalalignment='left', size='medium', color='red', weight='semibold')
    plt.text(0.1, 0.95, "easy-to-learn", horizontalalignment='left', size='medium', color='green', weight='semibold')
    plt.text(0.35, 0.5, "ambiguous", horizontalalignment='left', size='medium', color='blue', weight='semibold')

    plt.savefig(os.path.join(folder, filename))  
    

def save_transfer_log(transfer_info, path , train2val=True):
    if train2val:
        filename = "train2val.txt"
    else : 
        filename = "val2train.txt"
    
    file_path = os.path.join(path, filename)
   
    
    file_content = [f"{info}\n" for info in transfer_info]
    
    transfer_file = open(file_path, "w")
    transfer_file.writelines(file_content)
    
    transfer_file.close()
    
def save_crop_log(crop_img, crop_pos, path, fr_train=True):
    if fr_train:
        filename = "gen_train.txt"
    else: 
        filename = "gen_val.txt"
        
     
    file_path = os.path.join(path, filename)
    
    file_content = []
    
    for img, pos in zip(crop_img, crop_pos):
        file_content.append(f"{img} {pos}\n")
        
    
    crop_file = open(file_path, "w")
    crop_file.writelines(file_content)
    
    crop_file.close()
    
def save_img(img, filename, dir):
    file_path = os.path.join(dir, filename)
    
    cv2.imwrite(file_path, img)
    

def save_txt(contents, filename, dir):
    file_path = os.path.join(dir, filename)
    
    with open(file_path, 'w') as f : 
        for content in contents : 
            f.write(f"{content}\n")
            
    f.close()
    
    

def save_csv(img_name, pos, conf, cor, iou, path):

    # open the file in the write mode
    f = open(path, 'w')

    # create the csv writer
    writer = csv.writer(f)

    writer.writerow(['name', 'pos', 'confidence', 'correctness', 'iou'])

    for x in range(len(img_name)):
        writer.writerow([img_name[x], pos[x], conf[x], cor[x], iou[x]])
    # write a row to the csv file

    # close the file
    f.close()