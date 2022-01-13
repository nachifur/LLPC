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
from scipy.misc import imread
from utils import imshow, imshow_img_point
from PIL import Image


def gen_edge_from_point_base_gradient(data_path, debug):
    print("****** generate edge from point ->START ******\n")
    global edge_path, jsons_path, img_path, debug_g
    debug_g = debug
    jsons_path = os.path.join(data_path, 'json')
    edge_path = os.path.join(data_path, 'edge')
    create_dir(edge_path)
    f_jsons = sorted(os.listdir(jsons_path))
    img_path = os.path.join(data_path, 'png')

    if debug_g:
        # debug
        M = Multiprocessing(1)
        M.process(gen_edge_base_gradient, f_jsons[0:3])
    else:
        # run
        M = Multiprocessing(multiprocessing.cpu_count())
        M.process(gen_edge_base_gradient, f_jsons)

    M.close()
    print("\n****** generate edge from point ->END ******\n")


def gen_edge_base_gradient(f_img):
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    global edge_path, img_path, jsons_path, debug_g
    f_img_path = os.path.join(img_path, f_img.split('.')[0]+".png")
    f_json_path = os.path.join(jsons_path, f_img)
    data_id = f_img.split('.')[0]
    # load gradient
    img = imread(f_img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    gradient = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    gradient_GaussianBlur = cv2.GaussianBlur(gradient, (3, 3), 0)

    # load edge point
    with open(f_json_path) as f_obj:
        json_datas = json.load(f_obj)
    imH = json_datas['imageHeight']
    imW = json_datas['imageWidth']
    edge = np.zeros([imH, imW])
    cell_kernel_datas = json_datas['shapes']
    point_a = []
    point_edit_0 = []
    point_edit = []
    point_s = []
    search_radius = 7
    spacing = 1
    edge_width = np.sqrt(2)/2

    # edit edge based on gradient
    for cell_kernel_data in cell_kernel_datas:
        if cell_kernel_data['label'] in ['cell', 'kernel']:
            points = cell_kernel_data['points']
            points = np.array(points)

            y = points[:, 1]
            x = points[:, 0]

            group_num = 5  # 3,5,7...
            # filter label noise
            if group_num < len(x):
                point_s.append([x[:], y[:]])

                # edit edge based on gradient
                grad_v_suppress = 20  # gradient suppression value:25-35
                x_edit_0, y_edit_0, x_a, y_a = edit_edge_base_grad(
                    x, group_num, y, search_radius, imW, imH, spacing, edge_width, gradient_GaussianBlur, grad_v_suppress)

                # Linear interpolation where the spacing is large
                interp_step = 1  # 1<=interp_step<3. The smaller the value, the finer the fit
                x_edit_0, y_edit_0 = interp_point(
                    x_edit_0, y_edit_0, interp_step)

                # Use local weighted linear fitting to smooth edges, point->edges
                # 1.delete_end_point=-1, space=1 -> smooth edges, the sampling point interval can be adjusted by interp_step.
                # 2.delete_end_point=1,2,etc, space!=1, the sampling point interval can be adjusted by sample_step.
                # cell_group_num is odd, (cell_group_num-1)/2>delete_end_point
                cell_group_num = 7
                x_edit_1, y_edit_1 = local_linear_fit_edge(
                    x_edit_0, y_edit_0, cell_kernel_data['label'], interp_step, cell_group_num, delete_end_point=-1, sample_step=1)

                # Keep the edges continuous
                interp_step = 0.5
                x_edit_2, y_edit_2 = interp_point(
                    x_edit_1, y_edit_1, interp_step)

        point_edit_0.append([x_edit_0, y_edit_0])
        point_a.append([x_a, y_a])
        x_edit_2, y_edit_2 = limit_xy(imW, x_edit_2, imH, y_edit_2)
        point_edit.append([x_edit_2, y_edit_2])

    if debug_g:
        gradient_img = Image.fromarray(gradient)
        # candidate points, original points, smooth edge, discrete edge digital image
        # imshow_img_point(gradient_img, [[], [], [], point_edit])
        imshow_img_point(gradient_img, [point_a, point_s, point_edit_0, point_edit])

        sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        gradient = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        for point in point_edit:
            edge[(point[1]).astype(int),(point[0]).astype(int)] = uint8(255)
        edge[edge > 0] = 1
        edge = (morphology.skeletonize(edge)).astype(np.uint8)
        edge[edge == 1] = 1
        edge = edge[:, :, newaxis]
        imshow(Image.fromarray((1-edge)*(255-gradient)))
    else:
        # genarte edge image from point
        for point in point_edit:
            edge[(point[1]).astype(int),(point[0]).astype(int)] = uint8(255)
        edge[edge > 0] = 1
        edge = (morphology.skeletonize(edge)).astype(np.uint8)
        edge[edge == 1] = 255
        # save edge image
        cv2.imwrite(os.path.join(edge_path, data_id+'_edge.png'), edge)
        print('save ' + os.path.join(edge_path, data_id+'_edge.png'))


def limit_xy(imW, x_edit, imH, y_edit):
    x_edit[x_edit > imW-1] = imW-1
    x_edit[x_edit < 0] = 0
    y_edit[y_edit > imH-1] = imH-1
    y_edit[y_edit < 0] = 0
    # x_edit = np.rint(x_edit)
    # y_edit = np.rint(y_edit)
    # x_edit = x_edit.astype(int)
    # y_edit = y_edit.astype(int)
    return x_edit, y_edit


def local_linear_fit_edge(x_edit_0, y_edit_0, cell_kernel, interp_step, cell_group_num, delete_end_point=-1, sample_step=1, Weight_speed=1):
    # Use local weighted linear fitting to smooth edges, point->edges
    # select bandwidth (h): McCrary, Justin. 2008. “Manipulation of the runningvariable in the regression discontinuity design:A density test.” Journal of Econometrics 142 (2):698–714.
    #                       Ghanem D, Zhang J. ‘Effortless Perfection:’Do Chinese cities manipulate air pollution data?[J]. Journal of Environmental Economics and Management, 2014, 68(2): 203-225.
    point_num = len(x_edit_0)

    if cell_kernel == 'cell':
        group_num = int(interp_step*point_num/40) if int(interp_step *
                                                         point_num/40) > cell_group_num else cell_group_num
        a = 10
    else:
        group_num = int(interp_step*point_num /
                        10) if int(interp_step*point_num/10) > 3 else 3
        a = 5
    c = int(group_num*interp_step)/6

    fit_radius = int((group_num-1)/2)
    flag = 0
    if delete_end_point == -1:
        flag = 1
        delete_end_point = fit_radius-0.5
    if cell_kernel == 'kernel':
        delete_end_point = 0
    group_spacing = int((fit_radius-delete_end_point)*2)

    num_gap = point_num  # cyclic: num_gap = point_num, acyclic: num_gap = point_num-1
    num_all = (num_gap//group_spacing+1)*group_spacing-1
    for_l = np.arange(0, num_all, group_spacing)
    num_repeat_block = len(for_l)*group_spacing - num_gap

    x_edit_1 = np.array([])
    y_edit_1 = np.array([])
    for i in for_l:
        x_i = np.array([])
        y_i = np.array([])
        if i-fit_radius < 0:
            x_i = x_edit_0[i:i+fit_radius+1]
            y_i = y_edit_0[i:i+fit_radius+1]
            x_i = np.append(x_edit_0[0:i], x_i)
            y_i = np.append(y_edit_0[0:i], y_i)
            if len(x_i)-group_num < 0:
                x_i = np.append(x_edit_0[len(x_i)-group_num:], x_i)
                y_i = np.append(y_edit_0[len(y_i)-group_num:], y_i)
        else:
            if i+fit_radius+1 <= point_num:
                x_i = x_edit_0[i-fit_radius:i+fit_radius+1]
                y_i = y_edit_0[i-fit_radius:i+fit_radius+1]
            else:
                x_i = x_edit_0[i-fit_radius:]
                y_i = y_edit_0[i-fit_radius:]
                x_i = np.append(x_i, x_edit_0[0:group_num-len(x_i)])
                y_i = np.append(y_i, y_edit_0[0:group_num-len(y_i)])
        x_i = np.around(x_i).astype(int)
        y_i = np.around(y_i).astype(int)

        # rotating coordinate system
        R_theta = np.sqrt((x_i[-1]-x_i[0])**2+(y_i[-1]-y_i[0])**2)
        cos_theta = (x_i[-1]-x_i[0]) / \
            (R_theta+(R_theta == 0).astype(int))
        sin_theta = (y_i[-1]-y_i[0]) / \
            (R_theta+(R_theta == 0).astype(int))
        M_p = np.mat([[1, 0, 0],
                      [0, 1, 0],
                      [-x_i[0], -y_i[0], 1]]).T
        M_r = np.mat([[cos_theta, sin_theta, 0],
                      [-sin_theta, cos_theta, 0],
                      [0, 0, 1]])
        M = M_r*M_p
        if np.linalg.det(M) == 0:
            x_edit_1 = np.append(x_edit_1, x_i)
            y_edit_1 = np.append(y_edit_1, y_i)
        else:
            point_mat = np.mat(np.ones((3, len(x_i))))
            point_mat[0, :] = x_i[np.newaxis, :]
            point_mat[1, :] = y_i[np.newaxis, :]
            point_new_mat = M*point_mat
            if i == for_l[-1] and num_repeat_block != 0 and flag == 0:
                sample_x = point_new_mat[0, 0:-num_repeat_block]
                sample_y = point_new_mat[1, 0:-num_repeat_block]
            else:
                sample_x = point_new_mat[0, :]
                sample_y = point_new_mat[1, :]
            num_x = sample_x.shape[1]

            # edit sample_x -> Monotone sequence of numbers
            max_id = np.argmax(sample_x[0, :])
            exit_f = max_id != num_x-1
            if exit_f:
                sample_x = np.mat(np.linspace(
                    0, sample_x[0, -1], num_x))

            # local weighted linear fitting
            if flag == 0:
                fit_x = np.arange(
                    sample_x[0, delete_end_point], sample_x[0, -1-delete_end_point], sample_step)
            else:
                fit_x = np.array([sample_x[0, fit_radius+1]])

            b = 2*np.std(sample_y)/np.sqrt(num_x)
            h = a*b + c
            if h == 0:
                h = 1

            fit_y = np.zeros_like(fit_x)
            temp = np.mat(np.ones((2, num_x)))
            temp[1, :] = sample_x
            sample_x = temp
            for k_fit_y in range(0, len(fit_x)):
                w = np.mat(np.zeros((num_x, num_x)))
                K_h_all = np.zeros((num_x, 1))
                # compute K_h
                for k_w in range(0, num_x):
                    K_h_all[k_w] = gaussian_kernel(
                        (fit_x[k_fit_y]-sample_x[1, k_w])/h)/h
                sum_K_h_all = sum(K_h_all)
                if flag == 1:
                    max_k_h = np.max(K_h_all)
                    # K_h correction
                    for k_w in range(0, num_x):
                        if k_w == 0+delete_end_point or k_w == num_x-1-delete_end_point:
                            # Keep the edges continuous
                            K_h_all[k_w] = K_h_all[k_w]+max_k_h*1.5
                    sum_K_h_all = sum(K_h_all)
                # compute w  
                for k_w in range(0, num_x):
                    w[k_w, k_w] = K_h_all[k_w] / \
                        (sum_K_h_all+float(sum_K_h_all == 0))
                if np.linalg.det(sample_x*w*sample_x.T) == 0:
                    w = np.mat(np.identity(num_x))
                local_beta = ((sample_x*w*sample_x.T).I) * \
                    (sample_x*w*sample_y.T)
                fit_y[k_fit_y] = local_beta[0] + \
                    local_beta[1]*fit_x[k_fit_y]
            point_sample = np.mat(np.ones((3, len(fit_x))))
            point_sample[0, :] = fit_x[np.newaxis, :]
            point_sample[1, :] = fit_y[np.newaxis, :]
            point_sample = M.I*point_sample

            x_edit_1 = np.append(x_edit_1, np.array(
                point_sample[0, :]))
            y_edit_1 = np.append(y_edit_1, np.array(
                point_sample[1, :]))

            # plot
            # x_edit_1 = np.append(x_edit_1,np.array(point_sample[0,:]))
            # y_edit_1 = np.append(y_edit_1,np.array(point_sample[1,:]))
    return x_edit_1, y_edit_1


def interp_point(x_edit_0, y_edit_0, interp_step, cyclic=True):
    # Linear interpolation where the spacing is large
    x_edit_0 = x_edit_0.astype(float)
    y_edit_0 = y_edit_0.astype(float)
    if cyclic:
        dis = np.sqrt((x_edit_0[:]-np.append(x_edit_0[1:], x_edit_0[0]))
                      ** 2+(y_edit_0[:]-np.append(y_edit_0[1:], y_edit_0[0]))**2)
    else:
        dis = np.sqrt((x_edit_0[1:]-x_edit_0[:-1])
                      ** 2+(y_edit_0[1:]-y_edit_0[:-1])**2)
    id_e_l = dis > interp_step*2
    split_id = np.arange(0, len(id_e_l))[id_e_l]+1
    x_list = np.split(x_edit_0, split_id)
    y_list = np.split(y_edit_0, split_id)
    if len(x_list[-1]) != 0:
        x_list.append([])
        y_list.append([])

    x_extend = []
    y_extend = []
    for id in range(len(split_id)):
        if split_id[id] == len(x_edit_0):
            if cyclic:
                step = interp_step if x_edit_0[0] > x_edit_0[-1] else -interp_step
                x_e = np.arange(x_edit_0[-1]+step, x_edit_0[0], step)
                step = interp_step if y_edit_0[0] > y_edit_0[-1] else -interp_step
                y_e = np.arange(y_edit_0[-1]+step, y_edit_0[0], step)
                if len(x_e) > len(y_e):
                    if x_edit_0[0] > x_edit_0[-1]:
                        l_1 = [x_edit_0[-1], x_edit_0[0]]
                        l_2 = [y_edit_0[-1], y_edit_0[0]]
                    else:
                        l_1 = [x_edit_0[0], x_edit_0[-1]]
                        l_2 = [y_edit_0[0], y_edit_0[-1]]
                    y_e = np.interp(x_e, l_1, l_2)
                elif len(x_e) < len(y_e):
                    if y_edit_0[0] > y_edit_0[-1]:
                        l_1 = [x_edit_0[-1], x_edit_0[0]]
                        l_2 = [y_edit_0[-1], y_edit_0[0]]
                    else:
                        l_1 = [x_edit_0[0], x_edit_0[-1]]
                        l_2 = [y_edit_0[0], y_edit_0[-1]]
                    x_e = np.interp(y_e, l_2, l_1)
        else:
            step = interp_step if x_edit_0[split_id[id]
                                           ] > x_edit_0[split_id[id]-1] else -interp_step
            x_e = np.arange(
                x_edit_0[split_id[id]-1]+step, x_edit_0[split_id[id]], step)
            step = interp_step if y_edit_0[split_id[id]
                                           ] > y_edit_0[split_id[id]-1] else -interp_step
            y_e = np.arange(
                y_edit_0[split_id[id]-1]+step, y_edit_0[split_id[id]], step)
            if len(x_e) > len(y_e):
                if x_edit_0[split_id[id]] > x_edit_0[split_id[id]-1]:
                    l_1 = [x_edit_0[split_id[id]-1], x_edit_0[split_id[id]]]
                    l_2 = [y_edit_0[split_id[id]-1], y_edit_0[split_id[id]]]
                else:
                    l_1 = [x_edit_0[split_id[id]], x_edit_0[split_id[id]-1]]
                    l_2 = [y_edit_0[split_id[id]], y_edit_0[split_id[id]-1]]
                y_e = np.interp(x_e, l_1, l_2)
            elif len(x_e) < len(y_e):
                if y_edit_0[split_id[id]] > y_edit_0[split_id[id]-1]:
                    l_1 = [x_edit_0[split_id[id]-1], x_edit_0[split_id[id]]]
                    l_2 = [y_edit_0[split_id[id]-1], y_edit_0[split_id[id]]]
                else:
                    l_1 = [x_edit_0[split_id[id]], x_edit_0[split_id[id]-1]]
                    l_2 = [y_edit_0[split_id[id]], y_edit_0[split_id[id]-1]]
                x_e = np.interp(y_e, l_2, l_1)

        x_extend.append(x_e.astype(float))
        y_extend.append(y_e.astype(float))
    if len(x_extend) == 1:
        x_edit_0_list = np.insert(x_list, 1, [x_extend, []])
        y_edit_0_list = np.insert(y_list, 1, [y_extend, []])
    else:
        x_edit_0_list = np.insert(
            x_list, np.arange(0, len(x_extend))+1, x_extend)
        y_edit_0_list = np.insert(
            y_list, np.arange(0, len(x_extend))+1, y_extend)
    x_edit_0 = np.array([])
    for x_edit_0_ in x_edit_0_list:
        if len(x_edit_0_) != 0:
            x_edit_0 = np.append(x_edit_0, x_edit_0_)
    y_edit_0 = np.array([])
    for y_edit_0_ in y_edit_0_list:
        if len(y_edit_0_) != 0:
            y_edit_0 = np.append(y_edit_0, y_edit_0_)
    return x_edit_0, y_edit_0


def edit_edge_base_grad(x, group_num, y, search_radius, imW, imH, spacing, edge_width, gradient_GaussianBlur, grad_v_suppress):
    # edit edge based on gradient
    x_edit_0 = np.array([])
    y_edit_0 = np.array([])
    x_a = np.array([])
    y_a = np.array([])
    for i in range(0, x.size):
        x_i = np.array([])
        y_i = np.array([])
        if i+group_num <= x.size:
            x_i = x[i:i+group_num]
            y_i = y[i:i+group_num]
        else:
            x_i = x[i:]
            y_i = y[i:]
            x_i = np.append(x_i, x[0:group_num-len(x_i)])
            y_i = np.append(y_i, y[0:group_num-len(y_i)])
        if len(x_i) != group_num:
            print("--------error--------")
        x_i = np.around(x_i).astype(int)
        y_i = np.around(y_i).astype(int)
        x_cur = x_i[int((group_num-1)/2)]
        y_cur = y_i[int((group_num-1)/2)]

        # search direction y=y_cur+k(x-x_cur)
        if y_i[-1] == y_i[0]:
            point_0 = [x_cur, y_cur-search_radius]
            point_1 = [x_cur, y_cur+search_radius]
        else:
            k = -(x_i[-1]-x_i[0])/(y_i[-1]-y_i[0])
            point_0 = [x_cur-search_radius, y_cur-k*search_radius]
            point_1 = [x_cur+search_radius, y_cur+k*search_radius]
        R_theta = np.sqrt((x_i[-1]-x_i[0])**2+(y_i[-1]-y_i[0])**2)
        cos_theta = np.abs(x_i[-1]-x_i[0])/(R_theta+(R_theta == 0).astype(int))
        sin_theta = np.abs(y_i[-1]-y_i[0])/(R_theta+(R_theta == 0).astype(int))
        r_x = int(search_radius*sin_theta)
        r_y = int(search_radius*cos_theta)

        # boundary limit
        x_min = x_cur - r_x
        x_min = 0 if x_min < 0 else x_min
        x_max = x_cur + r_x
        x_max = imW-1 if x_max > imW-1 else x_max
        y_min = y_cur - r_y
        y_min = 0 if y_min < 0 else y_min
        y_max = y_cur + r_y
        y_max = imH-1 if y_max > imH-1 else y_max

        # get max point
        xnew_i = np.array([]).astype(int)
        ynew_i = np.array([]).astype(int)
        for i_local_x in range(x_min, x_max+spacing):
            for i_local_y in range(y_min, y_max+spacing):
                if point_linear_dis(point_0, point_1, [i_local_x, i_local_y]) < edge_width:
                    xnew_i = np.append(xnew_i, [i_local_x])
                    ynew_i = np.append(ynew_i, [i_local_y])

        x_a = np.append(x_a, xnew_i)
        y_a = np.append(y_a, ynew_i)
        candidate_grad = gradient_GaussianBlur[ynew_i, xnew_i]
        dis_center = np.sqrt((xnew_i-x_cur)**2+(ynew_i-y_cur)**2)
        h = search_radius/2
        w = gaussian_kernel(dis_center/h, 1)/h
        candidate_grad = candidate_grad*w
        max_id = np.argmax(candidate_grad)
        min_id = np.argmin(candidate_grad)

        # to suppress noise, we get max point from gradient_GaussianBlur
        # if abs(int(gradient_GaussianBlur[y_min, x_min])-int(gradient_GaussianBlur[y_max, x_max])) > max(w)*grad_v_suppress:
        # if abs(int(gradient_GaussianBlur[ynew_i[max_id], xnew_i[max_id]])-int(gradient_GaussianBlur[ynew_i[min_id], xnew_i[min_id]])) > max(w)*grad_v_suppress:
        if abs(candidate_grad[max_id]-candidate_grad[min_id]) > max(w)*grad_v_suppress:
            x_edit_point = xnew_i[max_id]
            y_edit_point = ynew_i[max_id]
        else:
            x_edit_point = x_cur
            y_edit_point = y_cur

        x_edit_0 = np.append(x_edit_0, x_edit_point)
        y_edit_0 = np.append(y_edit_0, y_edit_point)
    return x_edit_0, y_edit_0, x_a, y_a


def gaussian_kernel(x, Weight_speed=1):
    k = (1/(np.sqrt(2*np.pi)))*np.exp(-(x**2)/(2*Weight_speed**2))
    return k
