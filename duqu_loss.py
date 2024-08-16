import os
import json
import tensorflow as tf
import pickle
#
# def get_loss_from_tfevent_file(tfevent_filename):
#     """
#
#     :param tfevent_filename: the name of one tfevent file
#     :return: loss (list)
#     """
#     loss_val_list = []
#     for event in tf.train.summary_iterator(tfevent_filename):
#         print('1')
#         print(event)
#         for value in event.summary.value:
#             print('2')
#             print(' value ', value )
#             # print(value.tag)
#             if value.HasField('simple_value'):
#                 if value.tag == "loss":
#                     # print(value.simple_value)
#                     loss_val_list.append(value.simple_value)
#
#     return loss_val_list
#
# if __name__ == '__main__':
#     tfevent_filename = 'D:\study\CODE\daan_nh1117\-1431distributed_log\\train_distributed_trans_1112_aug3\\events.out.tfevents.1669645870.LAPTOP-MFSDB910'
#     data_pkl_addr = 'D:\study\CODE\daan_nh1117\-1431distributed_log\\train_distributed_trans_1112_aug3\\losstest.pkl'
#     loss = get_loss_from_tfevent_file(tfevent_filename)
#     # # 写入pkl文件
#     # with open(data_pkl_addr, 'wb') as f:
#     #     pickle.dump(loss, f)

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print('ea.scalars.Keys()',ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    print(type(val[0]))
    print(val[0])
    return val


def draw_plt(val, val_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name)
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('step')
    plt.ylabel(val_name)
    plt.show()


if __name__ == "__main__":
    tensorboard_path =  'D:\study\CODE\daan_nh1117\\1431distributed_log\\train_distributed_trans_1112_aug3\\events.out.tfevents.1669723175.LAPTOP-MFSDB910'
    data_pkl_addr = 'D:\study\CODE\daan_nh1117\-1431distributed_log\\train_distributed_trans_1112_aug3\\losstest.pkl'
    tfevent_filename = 'D:\study\CODE\daan_nh1117\-1431distributed_log\\train_distributed_trans_1112_aug3\\events.out.tfevents.1669645870.LAPTOP-MFSDB910'

    val_name = 'epoch_train/loss' #ea.scalars.Keys() ['epoch_train/class_acc', 'epoch_train/loss', 'epoch_val/class_acc', 'epoch_val/loss']
    val = read_tensorboard_data(tensorboard_path, val_name)
    x=[i.step for i in val]
    y = [j.value for j in val]
    print('x',x)
    print('y',y)
    # draw_plt(val, val_name)

    # # 写入pkl文件
    #     # with open(data_pkl_addr, 'wb') as f:
    #     #     pickle.dump(loss, f)
