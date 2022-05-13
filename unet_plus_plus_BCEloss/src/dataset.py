import glob
import os
import random

import cv2
import numpy as np
import scipy
import torchvision.transforms.functional as F
from PIL import Image
from scipy.misc import imread
from skimage import morphology
from torchvision import transforms as T

import torch
from torch.utils.data import DataLoader
from .utils import imshow
import Augmentor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_truth_flist, augment=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.data = self.load_flist(flist)
        self.edge_truth_data = self.load_flist(edge_truth_flist)

        self.input_size_h = config.INPUT_SIZE_H
        self.input_size_w = config.INPUT_SIZE_W

        self.NETWORK = config.NETWORK

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size_h = self.input_size_h
        size_w = self.input_size_w

        # load image
        img = imread(self.data[index])

        # load edge
        edge_truth = self.load_edge(size_h, size_w, index)

        # augment data
        if self.augment:
            img, edge_truth, mask = self.data_augment(
                img, edge_truth, size_h, size_w)
        else:
            mask = 255*np.uint8(np.ones((size_h, size_w)))

        # resize/crop if needed
        imgh, imgw = img.shape[0:2]
        if not (size_h == imgh and size_w == imgw):
            img = self.resize(img, size_h, size_w)

        # load gradient
        gradient = self.load_gradient(img, index)
        # imshow(Image.fromarray(edge_truth))
        return self.to_tensor(img), self.to_tensor(gradient), self.to_tensor(edge_truth), self.to_tensor(mask)

    def data_augment(self, images, edges, imgh, imgw):
        images = Image.fromarray(np.uint8(np.asarray(images)), mode="RGB")
        mask = Image.fromarray(np.uint8(np.ones_like(edges)))
        edges = Image.fromarray(np.uint8(edges))

        if np.random.binomial(1, 0.8) > 0:
            brightness_factor = np.random.uniform(0.8, 1.2)
            images = F.adjust_brightness(images, brightness_factor)
        if np.random.binomial(1, 0.8) > 0:
            contrast_factor = np.random.uniform(0.5, 2)
            images = F.adjust_contrast(images, contrast_factor)
        if np.random.binomial(1, 0.8) > 0:
            hue_factor = np.random.uniform(-0.2, 0.2)
            images = F.adjust_hue(images, hue_factor)
        if np.random.binomial(1, 0.8) > 0:
            saturation_factor = np.random.uniform(0.8, 1.2)
            images = F.adjust_saturation(images, saturation_factor)

        if np.random.binomial(1, 0.8) > 0:
            angle = random.randint(-2, 2)
            translate = (random.randint(-int(imgw*0.1), int(imgw*0.1)),
                         random.randint(-int(imgh*0.1), int(imgh*0.1)))
            scale = np.random.uniform(0.9, 1.1)
            if scale < 1:  # scale:[0.9,1.1]
                scale = scale/2+0.5
            shear = random.randint(-2, 2)
            images = F.affine(images, angle, translate, scale,
                              shear, resample=Image.BICUBIC)
            edges = F.affine(edges, angle, translate, scale,
                             shear, resample=Image.BICUBIC)
            mask = F.affine(mask, angle, translate, scale,
                            shear, resample=Image.BICUBIC)

        if np.random.binomial(1, 0.5) > 0:
            images = F.hflip(images)
            edges = F.hflip(edges)
            mask = F.hflip(mask)

        if np.random.binomial(1, 0.5) > 0:
            images = F.vflip(images)
            edges = F.vflip(edges)
            mask = F.vflip(mask)
        images = np.asarray(images)
        edges = np.asarray(edges)
        mask = np.asarray(mask)

        # https://github.com/mdbloice/Augmentor
        images = [[images, edges, mask]]
        p = Augmentor.DataPipeline(images)
        p.random_distortion(1, 10, 10, 10)
        g = p.generator(batch_size=1)
        augmented_images = next(g)
        images = augmented_images[0][0]
        edges = augmented_images[0][1]
        mask = augmented_images[0][2]

        edges = edges.copy()
        edges[edges < 128] = 0
        edges[edges >= 128] = 1
        edges = (morphology.skeletonize(edges)).astype(np.uint8)

        edges[edges == 1] = 255
        edges = self.resize(edges, imgh, imgw)
        mask = self.resize(mask, imgh, imgw)

        # 1.not thin edge truth
        edges[edges > 0] = 255

        # 2.thin edge truth
        # edges[edges>0]=1
        # edges = (morphology.skeletonize(edges)).astype(np.uint8)
        # edges[edges==1] = 255

        edges = (255-edges)*mask
        return images, edges, mask*255

    def load_edge(self, imgh, imgw, index):
        edge_truth_ = imread(self.edge_truth_data[index])
        edge_truth = self.resize(edge_truth_, imgh, imgw)
        if self.augment:
            edge_truth[edge_truth > 0] = 255
            imgh_src, imgw_src = edge_truth_.shape[0:2]
            edge_truth = self.resize(edge_truth, imgh_src, imgw_src)

        else:
            # 1.not thin edge truth
            edge_truth[edge_truth > 0] = 255

            # 2.thin edge truth
            # edge_truth[edge_truth>0]=1
            # edge_truth = (morphology.skeletonize(edge_truth)).astype(np.uint8)
            # edge_truth[edge_truth==1] = 255

            edge_truth = 255-edge_truth
        return edge_truth

    def load_gradient(self, img, index):
        imgh, imgw = img.shape[0:2]
        sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        gradient = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        return gradient

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width):
        imgh, imgw = img.shape[0:2]
        img = scipy.misc.imresize(img, [height, width])
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + \
                    list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
