# coding=utf-8
from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pickle as pk
from matplotlib.pyplot import cm
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import *
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras import utils as np_utils
from keras import Sequential
from keras.layers import *
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score,roc_auc_score
import keras.backend as k







def getyuanshitsne(x_test,y_test):
    x_test1 = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    X_tsne = pca_tsne.fit_transform(x_test1)

    for i in range(X_tsne.shape[0]):
        if y_test[i] == 0:
            c = 'b'
            mkr = 'o'
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=c, marker=mkr, linewidths=0)
    for i in range(X_tsne.shape[0]):
        if y_test[i] == 1:
            c = 'r'
            mkr = 'x'
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=c, marker=mkr, linewidths=2)
    plt.title('yuanshi')
    plt.show()



# model_addr='D:\study\科研\操作\时纯姐程序\mt_nhsj/'
# model_addr='D:\时纯的文件\课题代码\power-cnn-master\sample361w6\single_nhsj/'
# sample_pkl = 'D:\时纯的文件\课题代码\power-cnn-master\sample361w6/sample361w6_nh9_ycusW.pkl'
sample_pkl = 'D:\study\科研\操作\时纯姐程序/data_xywustime_nh9.pkl'

f=open(sample_pkl,'rb')
samp=pk.load(f)
X=samp['X']
y=samp['y']
# test_size=0.1
# random_state=3
x_train, x_test_all, y_train, y_test_all = train_test_split(
    X, y, test_size=0.4, random_state=1)

x_test, x_ver, y_test, y_ver = train_test_split(
    x_test_all, y_test_all, test_size=0.5, random_state=4)

l=len(x_ver)
pca_tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

# conved_2(l,x_ver,y_ver)
# W1=pd.read_csv('D:\时纯的文件\课题代码\power-cnn-master\sample39/W2.csv',header=None,)
# ypro=W1[0].values

print("******")
getyuanshitsne(x_ver,y_ver)
