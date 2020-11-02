import os
import numpy as np
import cv2
import yaml
from shutil import copyfile
import multiprocessing
import matplotlib.pyplot as plt


def create_config(config_path):
    if not os.path.exists(config_path):
        copyfile('config.yml.example', config_path)


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def resave_config(config, config_path):
    with open(config_path, 'w') as f_obj:
        yaml.dump(config, f_obj)
    with open(config_path, 'r') as f_obj:
        config = yaml.load(f_obj, Loader=yaml.FullLoader)
    return config


def point_linear_dis(xy2, xy1, point_xy):
    linear = np.array([xy2[0]-xy1[0], xy2[1]-xy1[1]])
    point_linePoint = np.array([xy2[0]-point_xy[0], xy2[1]-point_xy[1]])
    array_temp = (float(point_linePoint.dot(linear)) /
                  linear.dot(linear))
    array_temp = linear.dot(array_temp)
    distance = np.sqrt(
        (point_linePoint - array_temp).dot(point_linePoint - array_temp))
    return distance


def set_flist_config(config_path, train_val_test_path, flag='data'):
    with open(config_path, 'r') as f_obj:
        config = yaml.load(f_obj, Loader=yaml.FullLoader)
    if flag == 'mask':
        config['TRAIN_MASK_FLIST'] = train_val_test_path[0]
        config['VAL_MASK_FLIST'] = train_val_test_path[1]
        config['TEST_MASK_FLIST'] = train_val_test_path[2]
    if flag == 'data':
        config['TRAIN_FLIST'] = train_val_test_path[0]
        config['VAL_FLIST'] = train_val_test_path[1]
        config['TEST_FLIST'] = train_val_test_path[2]
    if flag == 'edge':
        config['TRAIN_EDGE_FLIST'] = train_val_test_path[0]
        config['VAL_EDGE_FLIST'] = train_val_test_path[1]
        config['TEST_EDGE_FLIST'] = train_val_test_path[2]
    with open(config_path, 'w') as f_obj:
        yaml.dump(config, f_obj)


def data_split(split_num, LOAD_PATH, SAVE_PATH, key_name, id_img, flist_path, RGB=False):
    # Generate split_num^2 small images
    print("******* data split ->START ******\n")
    edge_flist_list = np.genfromtxt(LOAD_PATH, dtype=np.str, encoding='utf-8')
    if len(edge_flist_list.shape) == 0:
        edge_flist_list_ = np.array([])
        edge_flist_list = np.append(edge_flist_list_, edge_flist_list)
    id_img_last = id_img
    image_flist = []
    for img_path in edge_flist_list:
        if RGB:
            img = cv2.imread(img_path, 3)
        else:
            img = cv2.imread(img_path, 0)
    #     cv2.imshow('edge', img)
    #     k = cv2.waitKey(0)
        imh, imw = img.shape[0:2]
        interval = [imh//split_num, imw//split_num]
        start_lists = [[0, 0], [interval[0]//2, 0],
                       [0, interval[1]//2], [interval[0]//2, interval[1]//2]]
        for start_list in start_lists:
            row = list(range(start_list[0], imh+1, interval[0]))
            col = list(range(start_list[1], imw+1, interval[1]))
            for i_row in range(0, len(row)-1):
                for i_col in range(0, len(col)-1):
                    if RGB:
                        cv2.imwrite(SAVE_PATH+'/'+str(id_img)+'_'+key_name+'.png',
                                    img[row[i_row]:row[i_row+1], col[i_col]:col[i_col+1], :])
                    else:
                        cv2.imwrite(SAVE_PATH+'/'+str(id_img)+'_'+key_name+'.png',
                                    img[row[i_row]:row[i_row+1], col[i_col]:col[i_col+1]])
                    image_flist.append(
                        SAVE_PATH+'/'+str(id_img)+'_'+key_name+'.png')
                    id_img += 1
        print('save '+img_path + "'s "+' sub image:' +
              str(id_img_last)+'-'+str(id_img))
        id_img_last = id_img
    np.savetxt(flist_path, image_flist, fmt='%s')
    print("******* data split ->END ******\n")
    return id_img


class Multiprocessing():
    def __init__(self, nThreads):
        self.nThreads = nThreads
        self.pool = multiprocessing.Pool(processes=self.nThreads)

    def process(self, fun, flist):
        self.pool.map(fun, flist)

    def close(self):
        self.pool.close()
        self.pool.join()


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    if len(img.size) == 3:
        plt.imshow(img, interpolation='none')
        plt.show()
    else:
        plt.imshow(img, cmap='Greys_r')
        plt.show()


def imshow_img_point(img, points_all, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('on')
    color = ['b', 'r', 'y', 'g']
    i = 0
    for points_i in points_all:
        for points in points_i:
            plt.plot(points[0], points[1], '.'+color[i])
            if i >= 2:
                points[0] = np.append(points[0], points[0][0])
                points[1] = np.append(points[1], points[1][0])
                plt.plot(points[0], points[1], '-'+color[i])
        i += 1

    if len(img.size) == 3:
        plt.imshow(img, interpolation='none')
        plt.show()
    else:
        plt.imshow(img, cmap='Greys_r')
        plt.show()
