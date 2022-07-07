"""
2022 lED
data split
"""

import os
from sklearn.model_selection import train_test_split


import os
import cv2

def aug_for_bad():
    bad_path = 'data/led/bad'
    a = os.listdir(bad_path)
    for i in a:
        if i[-4:] != '.jpg':
            continue
        print(bad_path+'/'+i)
        img = cv2.imread(bad_path+'/'+i)
        cv2.imwrite( 'data/led/bad_aug/'+i, img)
        img_flip = cv2.flip(img, 1)
        cv2.imwrite( 'data/led/bad_aug/'+i[:-4]+'1.jpg', img_flip)
        img_transpose = cv2.transpose(img)
        cv2.imwrite('data/led/bad_aug/'+i[:-4]+'2.jpg', img_transpose)
        img_flip = cv2.flip(img_transpose, 1)
        cv2.imwrite('data/led/bad_aug/'+i[:-4]+'3.jpg', img_flip)
        img_flip = cv2.flip(img_transpose, -1)
        cv2.imwrite('data/led/bad_aug/'+i[:-4]+'4.jpg', img_flip)

def train_val_split(imgs,ratio_train=0.8, ratio_val=0.2):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train+ratio_val) == 1
    train_img, val_img = train_test_split(imgs, test_size=1-ratio_train, random_state=233)
    # ratio=ratio_val/(1-ratio_train)
    print("NUMS of train:val = {}:{}".format(len(train_img), len(val_img)))
    return train_img, val_img


if __name__ == '__main__':
    good_img_path = "data/led/good"
    bad_img_path = "data/led/bad_aug"
    
    aug_for_bad()

    good_imgs =  os.listdir(good_img_path)
    bad_imgs =  os.listdir(bad_img_path)

    train_good_imgs,  val_good_imgs = train_val_split(good_imgs, ratio_train=0.5, ratio_val=0.5)
    train_bad_imgs,  val_bad_imgs = train_val_split(bad_imgs)

    anno_dir = os.path.dirname(good_img_path)
    # import pdb;pdb.set_trace()
    with open(os.path.join(anno_dir, 'train.txt'),'w') as ft:
        for img in train_good_imgs:
            line = "good/" + img + " " + "0\n"
            ft.writelines(line)
        for img in train_bad_imgs:
            line = "bad_aug/" + img + " " + "1\n"
            ft.writelines(line)

    with open(os.path.join(anno_dir, 'val.txt'),'w') as fv:
        for img in val_good_imgs:
            line = "good/" + img + " " + "0\n"
            fv.writelines(line)
        for img in val_bad_imgs:
            line = "bad_aug/" + img + " " + "1\n"
            fv.writelines(line)




