import json
import numpy as np
from scipy import interpolate
import random
import os

from pylab import *
from PIL import Image

from utils import point_linear_dis, create_dir, Multiprocessing

from skimage import morphology
import cv2
import multiprocessing
from utils import imshow
from PIL import Image


def gen_edge_from_point(data_path, debug):
    print("****** generate edge from point ->START ******\n")
    global edge_path, jsons_path, img_path,debug_g
    debug_g = debug
    jsons_path = os.path.join(data_path, 'json')
    edge_path = os.path.join(data_path, 'edge')
    create_dir(edge_path)
    f_jsons = sorted(os.listdir(jsons_path))
    img_path = os.path.join(data_path, 'png')

    if debug_g:
        # debug
        M = Multiprocessing(1)
        M.process(gen_edge, f_jsons[0:3])
    else:
        # run
        M = Multiprocessing(multiprocessing.cpu_count())
        M.process(gen_edge, f_jsons)

    M.close()
    print("\n******generate edge from point ->END ******\n")


def gen_edge(f_img):
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    global edge_path, img_path, jsons_path, debug_g
    f_img_path = os.path.join(img_path, f_img.split('.')[0]+".png")
    f_json_path = os.path.join(jsons_path, f_img)
    data_id = f_img.split('.')[0]

    with open(f_json_path) as f_obj:
        json_datas = json.load(f_obj)

    imH = json_datas['imageHeight']
    imW = json_datas['imageWidth']
    cell_kernel_datas = json_datas['shapes']

    edge = np.zeros([imH, imW],dtype='uint8')
    for cell_kernel_data in cell_kernel_datas:
        if True:
            points = cell_kernel_data['points']
            points = np.array(points)

            y = points[:, 1]
            x = points[:, 0]
            group_num = 2
            spacing = 1
            xnew = np.array([])
            ynew = np.array([])
            method = 'linear'
            edge_width = 0.5
            # filter label noise
            if group_num < len(x):
                if group_num == 1:
                    raise Exception('group_num > 1')
                else:
                    group_spacing = group_num-1
                if x.size % group_num:
                    num_all = (x.size//group_num)*group_num+1
                else:
                    num_all = (x.size//group_num)*group_num
                # get edge
                for i in range(0, num_all, group_spacing):
                    # get the points needed for interpolation
                    x_i = np.array([])
                    y_i = np.array([])
                    if i+group_num <= x.size:
                        x_i = x[i:i+group_num]
                        y_i = y[i:i+group_num]
                        x_i_ = x[i:i+group_num]
                        y_i_ = y[i:i+group_num]
                    else:
                        x_i = x[i:]
                        y_i = y[i:]
                        x_i = np.append(x_i, x[0])
                        y_i = np.append(y_i, y[0])
                        x_i_ = x_i
                        y_i_ = y_i

                    x_i = np.around(x_i).astype(int)
                    y_i = np.around(y_i).astype(int)
                    x_i_ = np.around(x_i_).astype(int)
                    y_i_ = np.around(y_i_).astype(int)

                    # get new point
                    xnew_i = np.array([])
                    ynew_i = np.array([])

                    if len(set(x_i)) == x_i.size:
                        func = interpolate.interp1d(x_i, y_i, kind=method)
                        for i_local_x in range(min(x_i_), max(x_i_)+spacing):
                            for i_local_y in range(min(y_i_), max(y_i_)+spacing):
                                if point_linear_dis([x_i[0], y_i[0]], [x_i[1], y_i[1]], [i_local_x, i_local_y]) < edge_width:
                                    xnew_i = np.append(xnew_i, [i_local_x])
                                    ynew_i = np.append(ynew_i, [i_local_y])
                    else:
                        if len(set(y_i)) == y_i.size:
                            func = interpolate.interp1d(y_i, x_i, kind=method)
                            for i_local_x in range(min(x_i_), max(x_i_)+spacing):
                                for i_local_y in range(min(y_i_), max(y_i_)+spacing):
                                    if point_linear_dis([x_i[0], y_i[0]], [x_i[1], y_i[1]], [i_local_x, i_local_y]) < edge_width:
                                        xnew_i = np.append(xnew_i, [i_local_x])
                                        ynew_i = np.append(ynew_i, [i_local_y])
                        else:
                            xnew_i = np.append(xnew_i, [])
                            ynew_i = np.append(ynew_i, [])

                    xnew = np.append(xnew, xnew_i)
                    ynew = np.append(ynew, ynew_i)
            # edge fusion
            xnew[xnew > imW-1] = imW-1
            xnew[xnew < 0] = 0
            ynew[ynew > imH-1] = imH-1
            ynew[ynew < 0] = 0

            xnew = xnew.astype('int')
            ynew = ynew.astype('int')
            # edge = morphology.skeletonize(edge)
            edge[ynew, xnew] = 255
    
    if debug_g:
        img = cv2.imread(f_img_path,-1)
        sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        gradient = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        edge[edge == 255] = 1  
        edge=edge[:,:,newaxis]
        imshow(Image.fromarray((1-edge)*(255-gradient)))
    else:
        # save edge
        cv2.imwrite(os.path.join(edge_path, data_id+'_edge.png'), edge)
        print('save ' + edge_path+data_id+'_edge.png')
