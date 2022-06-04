import os
import cv2
import numpy as np
import random
import pandas as pd
from utils import *
from torch.utils import data



class Dataset(data.Dataset):
    def __init__(self,opt):
        '''
        读取点云数据的路径存为list
        分为两种点云数据格式，mat与npy，mat保存为S3DFM的文件结构，npy文件存储在同一文件夹内。
        mat : -->m1-1-->seq1_001.mat
        npy : -->m1-1.npy
        opt.pattern选择不为mat与npy则报错并退出程序
        '''
        self.pattern = opt.pattern
        self.framenum =opt.framenum
        self.phase = opt.phase
        self.datatype = opt.datatype
        self.train_root = opt.train_root
        self.val_root = opt.val_root
        self.factor = opt.factor
        self.train = pd.read_csv(opt.train_csv,header = None) 
        self.val = pd.read_csv(opt.val_csv,header = None) 
        if self.phase == 'test':
            self.test = []
            self.testpath = opt.testpath.format(opt.datatype.title())
            foldlist = os.listdir(self.testpath)
            for fold in foldlist:
                self.test.append(os.path.join(self.testpath,fold))
    
    def __getitem__(self,index):
        data = []

        if self.phase == 'train':
            index = random.randint(0,self.train.shape[0]-1)
            information = self.train.iloc[index].to_numpy()
            root = self.train_root.format(self.datatype.title())
        elif self.phase == 'val':
            index = random.randint(0,self.val.shape[0]-1)
            information = self.val.iloc[index].to_numpy()
            root = self.val_root.format(self.datatype.title())
        elif self.phase == 'test':
            information = os.listdir(self.test[index])
            root = self.test[index]
        else:
            print('The parameter phase is error')
            exit()

        for file in information:
            if self.phase == 'test':
                data.append(cv2.imread(os.path.join(root,file),cv2.IMREAD_GRAYSCALE))
            else:
                data.append(cv2.imread(os.path.join(root,self.datatype.lower()+'_'+file),cv2.IMREAD_GRAYSCALE))

        data = np.array(data)
        f, h, w = data.shape
        if (w-h) > 0:
            white = np.zeros([f,round((w-h)/2),w])
            data = np.concatenate((white, data, white),1)
        else:
            white = np.zeros([f,h,round((h-w)/2)])
            data = np.concatenate((white, data, white),2)
        data = self.normalize(data)
        data = torch.from_numpy(data).unsqueeze(1)
        if self.phase == 'test':return data,round((w-h)/2),self.test[index].split('/')[-1]
        return data    
    def __len__(self):
        if self.phase == 'train':
            return round(self.train.shape[0])
        elif self.phase == 'val':
            return round(self.val.shape[0])
        elif self.phase == 'test':
            return len(self.test)

    @staticmethod
    def normalize(data):
        #pts = pts.astype(np.double)
        data = (data)/255
        return data


        

