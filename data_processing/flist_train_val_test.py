"""divide the data of all folders in a directory into: training set, verification set, test set"""
import os
import numpy as np
import random

def gen_flist_train_val_test(flist,edge_num,train_val_test_path,train_val_test_ratio,SEED,id_list):
    random.seed(SEED)
    # get flist
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF','json'}
    images = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
    images = images[0:edge_num]
    # shuffle
    files_num = len(images)
    images = sorted(images)
    if len(id_list)==0:
        id_list = list(range(files_num))
        shuffle=True
    else:
        shuffle=False
    if shuffle:
        random.shuffle(id_list)
    images = np.array(images)[id_list]
    # save
    images_train_val_test = [[],[],[]]
    i_list = [0]
    sum_=0
    if sum(train_val_test_ratio)==10:
        for i in range(3):
            if train_val_test_ratio[i]>0:
                sum_ = sum_+np.int(np.floor(train_val_test_ratio[i]*files_num/10))
                if i==2:
                    i_list.append(files_num)
                else:
                    i_list.append(sum_)
                images_train_val_test[i] = images[i_list[i]:i_list[i+1]]
                # save
                np.savetxt(train_val_test_path[i], images_train_val_test[i], fmt='%s')
            else:
                sum_ = sum_+np.int(np.floor(train_val_test_ratio[i]*files_num/10))
                i_list.append(sum_)
    else:
        print('input train_val_test_ratio error!')
    return id_list

