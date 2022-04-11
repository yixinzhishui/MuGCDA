# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:05:46 2020

@author: DYP
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 16:24:41 2020

@author: DYP
"""
import os
import numpy as np
import cv2
from torchvision import transforms as pytorchtrans
from torch.utils.data import Dataset
import torch

            
            
class Bigdata(Dataset):
    def __init__(
            self,
            img_A_path: str,
            save_path: str,
            cut_size = 1280,
            edge_padding = 128,
            overlap = 0.1, 
            max_batch = 8,
            transform = None
    ):      
        self.img_A_path = img_A_path
        self.max_batch = max_batch

        self.dataset_A=cv2.imread(self.img_A_path, -1)       #打开文件
        self.im_height, self.im_width, _ = self.dataset_A.shape    #栅格矩阵的列数
        self.resdata = np.zeros((self.im_height, self.im_width), 'uint8')
        self.cut_list = []
        self.cut_size = self.check_cutsize(cut_size, self.im_height, self.im_width)
        self.overlap = overlap
        self.save_path = save_path
        self.edge_padding = edge_padding
        self.gencut_list()
        if transform is None:
            self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor(),
                                              pytorchtrans.Normalize([[0.29310304, 0.2636692, 0.20883968, 0.39742813]],
                                                                     [0.09101917, 0.09300123, 0.106814794, 0.1142725]), ])
        else:
            self.tfms = transform
        pass
    def check_cutsize(self, cut_size, im_height, im_width):
        im_h = (im_height // 128) * 128
        im_w = (im_width // 128) * 128

        if cut_size>im_h: cut_size = im_h
        if cut_size>im_w: cut_size = im_w
        return cut_size
    
    def getbatchsize(self):
        if len(self.cut_list)<self.max_batch:
            return len(self.cut_list)
        return self.max_batch
            
        
    def gencut_list(self):
        width_overlap = self.overlap
        height_overlap = self.overlap
        image_width = self.cut_size
        image_height = self.cut_size
        #计算横向坐标
        width_list=[]
        for i in range((int)(self.im_width//(image_width*(1-width_overlap)))):
            if self.im_width - (i*(image_width*(1-width_overlap)))>=image_width:
                width_list.append((int)(i*(image_width*(1-width_overlap))))
        width_list.append(self.im_width-image_width)
        #计算纵向坐标
        height_list = []
        for i in range((int)(self.im_height//(image_height*(1-height_overlap)))):
            if self.im_height - (i*(image_height*(1-height_overlap)))>=image_height:
                height_list.append((int)(i*(image_height*(1-height_overlap))))
        height_list.append(self.im_height-image_height)
        
        for im_x in width_list:
            for im_y in height_list:
                self.cut_list.append([im_x, im_y])
                
    
    def cutimg(self, dataset, im_x, im_y):
        padding = [0, 0, 0, 0]
        cut_rect = [im_x - self.edge_padding,
                    im_y - self.edge_padding, 
                    im_x + self.cut_size + self.edge_padding,
                    im_y + self.cut_size + self.edge_padding]
        if cut_rect[0]<0:
            padding[0] = - cut_rect[0]
            cut_rect[0] = 0
        if cut_rect[1]<0:
            padding[1] = - cut_rect[1]
            cut_rect[1] = 0
        if cut_rect[2]>=self.im_width:
            padding[2] = cut_rect[2] - self.im_width
            cut_rect[2] = self.im_width
        if cut_rect[3]>=self.im_height:
            padding[3] = cut_rect[3] - self.im_height
            cut_rect[3] = self.im_height
        
        cut_rect = [int(i) for i in cut_rect]
        padding = [int(i) for i in padding]

        im_data = dataset[cut_rect[1]:cut_rect[3], cut_rect[0]:cut_rect[2]]
        return im_data, (padding[0], padding[2], padding[1], padding[3])
    
    
    def __len__(self):
        return len(self.cut_list)
    
    def __getitem__(self, i):
        im_x, im_y = self.cut_list[i]
        im_data, pad = self.cutimg(self.dataset_A, im_x, im_y)
        #im_A_data, pad = self.cutimg(self.dataset_A, im_x,im_y)
        #im_B_data, pad = self.cutimg(self.dataset_B, im_x,im_y)
        
        #im_data = np.concatenate([im_A_data, im_B_data], -1)
        
        sample = dict(
            im_x =im_x,
            im_y = im_y,
            image=im_data)
        
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype("uint8")).float().unsqueeze(0)
        padding = torch.nn.ReflectionPad2d(pad)
        sample['image'] = padding(sample['image'])[0]
        return sample
    
    def union_res(self, imx, imy, res):
        self.resdata[imy:imy + self.cut_size, imx:imx + self.cut_size] = res[self.edge_padding:self.edge_padding + self.cut_size, self.edge_padding:self.edge_padding + self.cut_size]
    
    def writeimg(self):
        cv2.imwrite(self.save_path,self.resdata)

    

if __name__ == "__main__":
    root_path = r'C:\Data\Competition\Hawei_2021\data\test-big'
    img_A_path = os.path.join(root_path, 'A/1.tif')
    img_B_path = os.path.join(root_path, 'B/1.tif')
    save_apth = os.path.join(root_path, 'res.tif')
    datagen = Bigdata_change(img_A_path, img_B_path, save_apth)
    for data in datagen:
        pass
        # print(data['image'].shape)
        # print(data['image'].shape, data['im_x'], data['im_y'])
    
        
        
        
            
            
            
            
            
            
            