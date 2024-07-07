from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import cv2
import numpy as np
class AttributeDataCelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, stargan_selected_attrs):
        super(AttributeDataCelebA, self).__init__()
        self.data_path = data_path #设置存放CelebA数据集的路径
        self.attr_path = attr_path #设置存放属性的相关路径
        self.selected_attrs = selected_attrs #设置选中的属性
        self.stargan_selected_attrs = stargan_selected_attrs #设置StarGAN用到的选中的属性
        self.tf = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        #通过读取第一行获取属性
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str_)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int_)
        #设置CelebA数据集的分割界标
        self.dataset_split_one=2000
        self.dataset_split_two=2500
        self.data_mode=mode
        if mode == 'train':
            self.images = images[:self.dataset_split_one]
            self.labels = labels[:self.dataset_split_one]
        if mode == 'test':
            self.images = images[self.dataset_split_one:self.dataset_split_two]
            self.labels = labels[self.dataset_split_one:self.dataset_split_two]
        if mode == 'test_more':
            self.images = images[self.dataset_split_two:]
            self.labels = labels[self.dataset_split_two:] 
        self.length = len(self.images)
        # stargan
        self.attr2idx = {}
        self.idx2attr = {}
        self.celeba_dataset = []
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        #这里获取了每行的属性标签相关内容
        #lines[0]是总数据量
        #lines[1]是相关属性
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        #这里获取了属性文件中的全部的属性名字
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        #我这里设置根据三个不同的数据集分别获取相应的属性
        if(self.data_mode=='train'):
            lines = lines[2:2+self.dataset_split_one]
        elif(self.data_mode=='test'):
            lines = lines[2+self.dataset_split_one:2+self.dataset_split_two]
        elif(self.data_mode=='test_more'):
            lines = lines[2+self.dataset_split_two:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in self.stargan_selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            self.celeba_dataset.append([filename, label])
        print('Finished preprocessing the CelebA dataset for Using')

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        filename, label = self.celeba_dataset[index]

        return img, att, torch.FloatTensor(label)
        
    def __len__(self):
        return self.length