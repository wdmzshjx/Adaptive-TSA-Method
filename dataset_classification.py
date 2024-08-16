
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pickle
import pickle

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from augmentation import *
import random


def StandardScaler_X_train(x_train):  #归一化处理

    scaler = StandardScaler()
    i = 0
    for x in x_train:
        x = scaler.fit_transform(x)
        x_train[i,:,:] = x
        i += 1
    return x_train


from sklearn.model_selection import train_test_split

def dataset_classi(sample_pkl,input_x_key=None,input_y_key=None):
    sample = pickle.load(open(sample_pkl, 'rb'))
    print(sample.keys())
    X = sample[input_x_key]
    y = sample[input_y_key]


    from collections import Counter

    ylabel=y[:,0]
    class_dict_nums = dict(Counter(ylabel))
    print(class_dict_nums)
    weights_train =[class_dict_nums[0],class_dict_nums[1]]
    weights_train = [max(weights_train)/x for x in weights_train]
    print('失稳0样本个数：'+str(np.sum(ylabel==0)),'\n','稳定1样本个数：'+str(np.sum(ylabel==1)))

    X=StandardScaler_X_train(X)
    x_train, x_test_all, y_train, y_test_all = train_test_split(
        X, y, test_size=0.4, random_state=1)

    x_test, x_ver, y_test, y_ver = train_test_split(
        x_test_all, y_test_all, test_size=0.5, random_state=4)


    return x_train,y_train,x_test,y_test,x_ver,y_ver,x_test_all,y_test_all,weights_train

def dataset_classi_new(sample_pkl, sample_new_pkl, input_x_key=None, input_y_key=None):
    sample = pickle.load(open(sample_pkl, 'rb'))
    print(sample.keys())
    X = sample[input_x_key]
    y = sample[input_y_key]

    sample_new = pickle.load(open(sample_new_pkl, 'rb'))
    print(sample_new.keys())
    X_new = sample_new[input_x_key]
    y_new = sample_new[input_y_key]
    from collections import Counter

    ylabel = y[:, 0]
    class_dict_nums = dict(Counter(ylabel))
    weights_train = [class_dict_nums[0], class_dict_nums[1]]
    weights_train = [max(weights_train) / x for x in weights_train]
    print('失稳0样本个数：' + str(np.sum(ylabel == 0)), '\n', '稳定1样本个数：' + str(np.sum(ylabel == 1)))

    print('新增样本失稳样本0个数：' + str(np.sum(y_new[:, 0] == 0)), '\n', '稳定1样本个数：' + str(np.sum(y_new[:, 0] == 1)))

    X = StandardScaler_X_train(X)
    x_train, x_test_all, y_train, y_test_all = train_test_split(
        X, y, test_size=0.4, random_state=1)
    x_test, x_ver, y_test, y_ver = train_test_split(
        x_test_all, y_test_all, test_size=0.5, random_state=4)

    X_new = StandardScaler_X_train(X_new)
    x_train_new, x_test_all_new, y_train_new, y_test_all_new = train_test_split(X_new, y_new, test_size=0.6, random_state=1)  # 将边界样本全部保存在x_train_new中

    x_train_new1= np.concatenate((x_train_new,x_train), axis=0)
    y_train_new1 = np.concatenate((y_train_new, y_train), axis=0)
    x_train_new2, x_test_all2, y_train_new2, y_test_all2 = train_test_split(x_train_new1, y_train_new1, test_size=0.5, random_state=2)
    x_train_new2 = np.concatenate((x_train_new2,x_test_all2),axis=0)
    y_train_new2 = np.concatenate((y_train_new2,y_test_all2),axis=0)
    return x_train_new2, y_train_new2, x_test, y_test, x_ver, y_ver, x_test_all, y_test_all, weights_train


def random_change_to_(DATA,lendata,percent,changeto):
    for data in DATA:
        index = np.random.randint(lendata, size=int(lendata*percent/100))
        for i in index:
            m = int(i/50)
            n = int(i%50)
            data[m,n] = changeto
            # print(i,m,n,data[m,n])
    print('random_change_to_修改完成，lendata,percent,changeto：',lendata,percent,changeto)
    return DATA


class DatasetGenerate(Dataset):
    def __init__(self,x_train,y_train,data_type='train'):
        self.x_train = x_train
        self.y_train = y_train #note list时
        # self.y_train = y_train[:,0]
        self.data_type = data_type
    def __getitem__(self, idx): #根据index取列表中的数值
        # sample = self.samples[idx]
        # x_input = sample['x'] #x_input-list格式-6个特征|x_input[0]-list格式-50个点
        # x_input = np.array(x_input)

        label = self.y_train[idx]

        input_x = np.float32(self.x_train[idx:idx+1,:,:])
        x_input = input_x[0]

        debug_bool = False
        if self.data_type=='train' and False: #and False-》不做数据增强
            #####添加噪音
            # print(100*'=',np.random.uniform())
            if np.random.uniform() >0.95 or debug_bool:
                x_input = jitter(np.array([x_input]),sigma=0.03)[0]

            #####数据缩放
            if np.random.uniform() >0.95 or debug_bool: #以0.05的比例做增强
                # print(20*'=','x_input',x_input.shape)
                x_input = scaling(np.array([x_input]), sigma=0.1)[0]
            ###幅度翘曲
            if np.random.uniform() >0.95 or debug_bool:
                # print(20*'=','x_input',x_input.shape)
                x_input = magnitude_warp(np.array([x_input]), sigma=0.2, knot=4)[0]
            #####窗口切片
            if np.random.uniform() >0.95 or debug_bool:
                # print(20*'=','x_input window_slice',x_input.shape)
                x_input = window_slice(np.array([x_input]), reduce_ratio=0.9)[0]
            ####窗口翘曲
            if np.random.uniform() >0.95 or debug_bool:
                # print(20*'=','x_input',x_input.shape)
                x_input = window_warp(np.array([x_input]), window_ratio=0.1, scales=[2, 1.1])[0]

        if True:
            x_input = np.expand_dims(x_input,axis=0) #在axis=0的维度上增加一维#x_input.shape (1, 6, 50)

        # print(input_x.shape)
        # return torch.from_numpy(np.float32(x_input)),torch.tensor(int(label)) #note list时
        return torch.from_numpy(np.float32(x_input)), torch.tensor(label)  # label+index

    def __len__(self):
        return self.x_train.shape[0]






if __name__=='__main__':
    sample_pkl='6_qdys_rgb_3+7+5_315.pkl'
    input_x_key='DATARGB'
    input_y_key='WEN01'
    x_train,y_train,x_test,y_test,x_ver,y_ver,x_test_all,y_test_all,weights_train =  dataset_classi(sample_pkl,input_x_key=input_x_key,input_y_key=input_y_key)
    # print(x_train.shape)
    # print(y_train.shape)
    dataset = DatasetGenerate(x_train,y_train)
    x,y = dataset.__getitem__(0)

    print(x.size())
    print(y.size(),y)