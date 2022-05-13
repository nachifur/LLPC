from edge_linear import gen_edge_from_point
from edge_linear_base_gradient import gen_edge_from_point_base_gradient
from flist import gen_flist
from flist_train_val_test import gen_flist_train_val_test
from utils import create_dir, set_flist_config, data_split, resave_config, create_config
import yaml
import os
import numpy as np
import time
"""run"""
# cd main_path
# python scripts/data_processing.py


def data_processing(DATA_PATH, ratio_list, debug, label_correct=True):
    """configuration"""
    if label_correct:
        config_path = './label_correct_config.yml'  # config loadpath
    else:
        config_path = './label_no_correct_config.yml'  # config loadpath
    create_config(config_path)
    with open(config_path, 'r') as f_obj:
        config = yaml.load(f_obj, Loader=yaml.FullLoader)

    split = config['SPLIT']
    split_num = config['SPLIT_NUM']  # final split image number is split_num^2

    if split:
        DATA_SAVE_PATH = os.path.join(
            DATA_PATH, 'datasets_split')  # flist savepath
    else:
        DATA_SAVE_PATH = os.path.join(DATA_PATH + 'datasets')

    IMG_SPLIT_SAVE_PATH = os.path.join(
        DATA_PATH, 'png_split')  # img split savepath
    EDGE_SPLIT_SAVE_PATH = os.path.join(
        DATA_PATH, 'edge_split')  # edge split savepath

    JSON_SAVE_PATH = os.path.join(
        DATA_PATH, 'json_correct')  # .json correct savepath

    # save path
    create_dir(DATA_SAVE_PATH)
    create_dir(JSON_SAVE_PATH)
    if split:
        create_dir(IMG_SPLIT_SAVE_PATH)
        create_dir(EDGE_SPLIT_SAVE_PATH)

    ## generate edge from points
    # time_start=time.time()
    # print(time_start)
    # if label_correct:
    #     gen_edge_from_point_base_gradient(DATA_PATH, debug)
    # else:
    #     gen_edge_from_point(DATA_PATH, debug)
    # time_end=time.time()
    # print(time_end)
    # print('generate edge from points time cost',time_end-time_start,'s')
    
    if debug==0:
        subject_word = config['SUBJECT_WORD']

        # generate a list of original edge
        edge_flist_src = os.path.join(DATA_SAVE_PATH, subject_word + '_edge.flist')
        gen_flist(os.path.join(DATA_PATH, 'edge'), edge_flist_src)
        edge_num = len(np.genfromtxt(
            edge_flist_src, dtype=np.str, encoding='utf-8'))
        # generate a list of original images
        png_flist_src = os.path.join(DATA_SAVE_PATH, subject_word + '_png.flist')
        gen_flist(os.path.join(DATA_PATH, 'png'), png_flist_src)

        # img (training set, verification set, test set)(not split)
        key_name = 'png'
        png_flist = os.path.join(DATA_SAVE_PATH, subject_word + '_' + key_name)
        png_val_test_PATH = [png_flist+'_train.flist',
                            png_flist+'_val.flist', png_flist+'_test.flist']
        id_list = gen_flist_train_val_test(
            png_flist_src, edge_num, png_val_test_PATH, ratio_list, config['SEED'], [])
        # edge (training set, verification set, test set)(not split)
        key_name = 'edge'
        edge_flist = os.path.join(DATA_SAVE_PATH, subject_word + '_' + key_name)
        edge_val_test_PATH = [edge_flist+'_train.flist',
                            edge_flist+'_val.flist', edge_flist+'_test.flist']
        gen_flist_train_val_test(
            edge_flist_src, edge_num, edge_val_test_PATH, ratio_list, config['SEED'], id_list)

        # split data
        if split:
            key_name = 'png_split'
            png_flist = os.path.join(DATA_SAVE_PATH, subject_word + '_' + key_name)
            png_val_test_PATH_save = [png_flist+'_train.flist',
                                    png_flist+'_val.flist', png_flist+'_test.flist']
            i = 0
            id_img = 0
            for path in png_val_test_PATH:
                if ratio_list[i] != 0:
                    id_img = data_split(split_num, path, IMG_SPLIT_SAVE_PATH,
                                        'png', id_img, png_val_test_PATH_save[i], RGB=True)
                i += 1

            key_name = 'edge_split'
            png_flist = os.path.join(DATA_SAVE_PATH, subject_word + '_' + key_name)
            edge_val_test_PATH_save = [
                png_flist+'_train.flist', png_flist+'_val.flist', png_flist+'_test.flist']
            i = 0
            id_img = 0
            for path in edge_val_test_PATH:
                if ratio_list[i] != 0:
                    id_img = data_split(split_num, path, EDGE_SPLIT_SAVE_PATH,
                                        'edge', id_img, edge_val_test_PATH_save[i], RGB=False)
                i += 1

            png_val_test_PATH = png_val_test_PATH_save
            edge_val_test_PATH = edge_val_test_PATH_save

        """setting path of data list"""
        set_flist_config(config_path, png_val_test_PATH, flag='data')
        set_flist_config(config_path, edge_val_test_PATH, flag='edge')
