#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from PIL import Image
import numpy as np
from random import shuffle
from threadsafe_iter import threadsafe_generator

import image_processing
import cv2

class DataGenerator:
    def __init__(self,
                 dataset_path,
                 labelHsh_path,
                 batch_size,
                 image_size,
                 horizontal_flip = False,
                 vertical_flip = False,
                 rotation_range=0,
                 shear_range=0.,
                 ):
        self.index = 0
        self.load_data(dataset_path=dataset_path)
        self.labelHsh = np.load(labelHsh_path)
        self.batch_size = batch_size
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range =rotation_range
        self.shear_range = shear_range



    def load_data(self,dataset_path):
        with open(dataset_path,'r',encoding='utf-8') as f:
            self.dataset =f.readlines()
        shuffle(self.dataset)



    def get_data_number(self):
        return int(len(self.dataset))


    def load_data_labels(self):
        image_path , image_label =self.dataset[self.index].strip().split(' ')
        return image_path , self.labelHsh[int(image_label)]

    @threadsafe_generator
    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if (self.index == len(self.dataset)):
                    self.index = 0
                img_path ,img_label =self.load_data_labels()
                img = cv2.imread(img_path)
                img = cv2.resize(img, (self.image_size[0],self.image_size[1]), interpolation=cv2.INTER_LINEAR)
                img = np.array(img)

                # 数据增强
                img = image_processing.image_process(x_img=img, horizontal_flip=self.horizontal_flip,
                                                     vertical_flip=self.vertical_flip,
                                                     rotation_range=self.rotation_range,
                                                     shear_range=self.shear_range)
                batch_images.append(img)
                batch_labels.append(img_label)
                self.index += 1
            try:
                batch_images = np.array(batch_images)
                batch_labels = np.array(batch_labels)
            except Exception as e:
                print(e, '\n' + str(img_path) + ": 文件出错 \n")
                continue
            yield batch_images ,batch_labels


# if __name__ == '__main__':
#     TRAIN_TXT = r"train.txt"
#     VALID_TXT = r"valid.txt"
#     labelHsh_path = './Data/labelHsh24_8_254.npy'
#     temp=np.load(labelHsh_path)
#     IN_SIZE = (256, 256)
#     BATCH_SIZE=32
#
#     valid_gen = DataGenerator(dataset_path=VALID_TXT,
#                               labelHsh_path=labelHsh_path,
#                               batch_size=BATCH_SIZE,
#                               image_size=IN_SIZE,
#                               )
#     DataGenerator.load_data(valid_gen, 'valid.txt')
#     t,p=DataGenerator.load_data_labels(valid_gen)
#     x, y= next(valid_gen.get_mini_batch())