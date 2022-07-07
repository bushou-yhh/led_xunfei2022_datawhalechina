"""
xun fei 2022 lED 

pseudo label generator
"""

from cProfile import label
import os
from sklearn.model_selection import train_test_split


import os
import csv
import shutil


def train_val_split(imgs,ratio_train=0.8, ratio_val=0.2):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train+ratio_val) == 1
    train_img, val_img = train_test_split(imgs, test_size=1-ratio_train, random_state=233)
    # ratio=ratio_val/(1-ratio_train)
    print("NUMS of train:val = {}:{}".format(len(train_img), len(val_img)))
    return train_img, val_img


if __name__ == '__main__':
    result_path = "data/led/pu_result_swin-tiny1.csv"
    pseudolabel_path = "data/led"
    train_txt = "data/led/train.txt"
    pseudo_train_txt = "data/led/train_pseudo.txt"
    val_txt = 'data/led/val.txt'
    pseudo_val_txt = "data/led/val_pseudo.txt"

    datas = []
    with open(result_path, 'r') as f:
        csv_reader = csv.reader(f)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)        # 读取第一行每一列的标题
        for row in csv_reader: 
            datas.append(row)
            
           
    train_pseudos,  val_pseudos = train_val_split(datas)

    shutil.copyfile(train_txt, pseudo_train_txt)
    shutil.copyfile(val_txt, pseudo_val_txt)
    
    with open(pseudo_train_txt,'a') as ft:
        for  train_pseudo in train_pseudos:
            # import pdb;pdb.set_trace()
            img, label = train_pseudo
            line = "test/" + img + " " + label + '\n'
            ft.writelines(line)

    with open(pseudo_val_txt ,'a') as fv:
        for val_pseudo in val_pseudos:
            img, label = val_pseudo
            line = "test/" + img + " " + label + '\n'
            fv.writelines(line)




