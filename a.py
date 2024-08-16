
from dataset import read_pickle
import numpy as np 

pkl_path = '1_qdys_rgb_3+7+5_315/1_qdys_rgb_3+7+5_315.pkl'

pkl_path = '5_qdys-rgb_gjc-0.pkl'
pkl_path = '6_qdys_rgb_3+7+5_315.pkl'
pkl_path = '08_qdys_rgb_3+7+5_315.pkl'
data_list = read_pickle(pkl_path)
print(data_list[0].keys())
# print(data_list[0]['DATAQDYS'].shape)###(15294, 15, 50)
# print(data_list[0]['GJCQDYS'].shape) ###(15294, 15, 30)   ##################### 1110
print(data_list[0]['DATARGB'].shape)###(15294, 45, 50)    ##################### 1111
# print(data_list[0]['GJCRGB'].shape)###(15294, 45, 30)    ######################1112
print(len(data_list[0]['WEN01']))

print(np.min(data_list[0]['WEN01']))