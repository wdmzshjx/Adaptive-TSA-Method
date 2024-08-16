import torch.nn as nn
from torchvision import models
import torch
import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F


resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name,args):

    print(name)
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "classification1" == name.lower():
        return Classification1()
    elif "classification2" == name.lower():
        return Classification2()
    elif "classification3" == name.lower():
        # return Classification5()
        ####Classification5的模型效果可以
        #  ####EfficientNet.from_pretrained('efficientnet-own') ####Classification3
        # return Classification6()
        return Classification7(args.model_feature_num,args.model_time_num,args.feature_dim)
    elif "classification4" == name.lower():
        return Classification4()
    elif "classification7" ==name.lower():
        return Classification7()
    elif "classification8" == name.lower():
        return Classification8()
    elif "classification9" == name.lower():
        return Classification9()
    
class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim

class Classification1(nn.Module):
    def __init__(self):
        super(Classification1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6,out_channels=50,kernel_size=2)

        self.bn1 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU(inplace=True)
        self._feature_dim = 2450 #4950-》2450
        # self.fc = nn.Sequential(
        #     nn.Linear(4950, int(hidden_units)),
        #     nn.SELU(True),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(int(hidden_units), num_classes),
        # )

    def forward(self, x): #torch.Size([32, 6, 50])
        print(x.size())
        x = self.conv1(x) #torch.Size([32, 50, 49])
        print(x.size())
        x = self.bn1(x)#torch.Size([32, 50, 49])
        print(x.size())
        x = self.relu(x)#torch.Size([32, 50, 49])
        print(x.size())
        x = x.view(x.size(0), -1) #torch.Size([32, 2450])
        print(x.size())
        return x

    def output_num(self):
        return self._feature_dim


class Classification2(nn.Module):
    def __init__(self):
        super(Classification2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=6,out_channels=50,kernel_size=2),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=50,out_channels=50,kernel_size=2),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True)
        )
        self._feature_dim = 2400

    def forward(self, x):#torch.Size([32, 6, 50])
        # print(x.size())
        x = self.conv1(x)#torch.Size([32, 50, 49])
        # print(x.size())
        x = self.conv2(x) #torch.Size([32, 50, 48])
        # print(x.size())
        x = x.view(x.size(0), -1) #torch.Size([32, 2400])
        # print(x.size())

        return x

    def output_num(self):
        return self._feature_dim


class Classification3(nn.Module):
    def __init__(self):
        super(Classification3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=1,stride=1), # ,padding='same' |kernel_size=3-》1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1), #,padding='same' | kernel_size=3-》2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._feature_dim = 2688


    def forward(self, x): #torch.Size([32, 1, 31, 12])
        # print(x.size())
        x = self.conv1(x) #torch.Size([32, 32, 31, 12])/1
        # print(x.size())
        x = self.pool1(x) #torch.Size([32, 32, 15, 12])/2
        # print(x.size())
        x = self.conv2(x) #torch.Size([32, 64, 15, 12])/1
        # print(x.size())
        x = self.pool2(x) #torch.Size([32, 64, 7, 6])/2
        # print(x.size())
        x = x.view(x.size(0), -1) #torch.Size([32, 2688])
        # print(x.size())

        return x

    def output_num(self):
        return self._feature_dim

class Classification4(nn.Module):
    def __init__(self):
        super(Classification4, self).__init__()

        self.lstm1 = nn.LSTM(50, 32, 1, batch_first=True, bidirectional=True) #0为100-》50
        self.lstm2 = nn.LSTM(64, 32, 1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(64, 32, 1, batch_first=True,)
        self._feature_dim = 192


    def forward(self, x): #torch.Size([32, 6, 50])
        print(x.size())
        x,_ = self.lstm1(x) #torch.Size([32, 6, 64])
        print(x.size())
        x,_ = self.lstm2(x) #torch.Size([32, 6, 64])
        print(x.size())
        x,_ = self.lstm3(x) #torch.Size([32, 6, 32])
        print(x.size())
        x = x.contiguous().view(x.size(0), -1) #torch.Size([32, 192])
        print(x.size())

        return x

    def output_num(self):
        return self._feature_dim






class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        # print('x.size()',x.size())
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
            # print('(pad_w // 2, pad_w - pad_w // 2,pad_h // 2, pad_h - pad_h // 2)',(pad_w // 2, pad_w - pad_w // 2,pad_h // 2, pad_h - pad_h // 2))
        else:
            # print("!!!!!!!!pad_h = 0 and pad_w = 0")
            self.static_padding = nn.Identity()

    def forward(self, x):
        # print('x.size',x.size()) #torch.Size([batchsize2000, 输入行-时间点数9, 输入列-特征数50])
        x = self.static_padding(x)
        # print('!!!!!!!!!!!!!x.size()',x.size())
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print('!!!!!!!!!!!!!x.size()!!!!!!!!!', x.size())
        return x



def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Classification5(nn.Module):
    def __init__(self):
        super(Classification5, self).__init__()

        self.conv1 = nn.Sequential(
            get_same_padding_conv2d(image_size=(31,12))(in_channels=1,out_channels=16, kernel_size=1, bias=False), # ,padding='same' |kernel_size=3-》1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self._swish = MemoryEfficientSwish()
        self._feature_dim = 5952


    def forward(self, x): #torch.Size([32, 1, 31, 12])
        x = self.conv1(x) #torch.Size([32, 32, 31, 12])/1
        x = self._swish(x)

        x = x.view(x.size(0), -1) #torch.Size([32, 2688])


        return x

    def output_num(self):
        return self._feature_dim



class Classification6(nn.Module):
    def __init__(self):
        super(Classification6, self).__init__()
        self.lstm1 = nn.LSTM(12, 32, 2, batch_first=True, bidirectional=True) #0为100-》50
        self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True,)
        self._feature_dim = 992


    def forward(self, xx): #torch.Size([32, 6, 50])
        # print(x.size())
        x = xx.squeeze(1)
        x,_ = self.lstm1(x) #torch.Size([32, 6, 64])
        x,_ = self.lstm2(x) #torch.Size([32, 6, 64])
        x,_ = self.lstm3(x) #torch.Size([32, 6, 32])
        x = x.contiguous().view(x.size(0), -1) #torch.Size([32, 192])

        return x

    def output_num(self):
        return self._feature_dim

'''正常的trans使用'''
class Classification7(nn.Module):
    def __init__(self,model_feature_num=50,model_time_num=9,feature_dim=7488): #model_feature_num=12,model_time_num=31，feature_dim=8160-最后的x的值
        super(Classification7, self).__init__()


        self.conv1 = nn.Sequential(
            get_same_padding_conv2d(image_size=(model_time_num,model_feature_num))(in_channels=1,out_channels=16, kernel_size=1, bias=False), # ,padding='same' |kernel_size=3-》1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self._swish = MemoryEfficientSwish()

        self.lstm1 = nn.LSTM(model_feature_num, 32, 2, batch_first=True, bidirectional=True) #0为100-》50
        self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True,)
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        self.out_features = feature_dim

    def forward(self, xx): #torch.Size([32, 6, 50])
        # print(x.size())

        covx = self.conv1(xx) #torch.Size([32, 32, 31, 12])/1
        covx = self._swish(covx)

        covx = covx.view(covx.size(0), -1) #torch.Size([32, 2688])


        x = xx.squeeze(1)
        x,_ = self.lstm1(x) #torch.Size([32, 6, 64])
        x,_ = self.lstm2(x) #torch.Size([32, 6, 64])
        x,_ = self.lstm3(x) #torch.Size([32, 6, 32])
        x = x.contiguous().view(x.size(0), -1) #torch.Size([32, 192])
        # print('x',x.size(),'covx',covx.size())
        x = torch.cat([x,covx],1)
        # print('最后x',x.size(),'covx',covx.size()) #9*50 最后x torch.Size([2000, 7488]) covx torch.Size([2000, 7200])
        print('正常的训练classification7')

        return x

    def output_num(self):
        return self._feature_dim

# '''加drop的trans使用'''
# class Classification7(nn.Module):
#     def __init__(self, model_feature_num=50, model_time_num=9,
#                  feature_dim=7488):  # model_feature_num=12,model_time_num=31，feature_dim=8160-最后的x的值
#         super(Classification7, self).__init__()
#
#         drop = 0.1
#         self.conv1 = nn.Sequential(
#             get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
#                                                                                     kernel_size=1, bias=False),
#             # ,padding='same' |kernel_size=3-》1
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=drop)
#         )
#
#         self._swish = MemoryEfficientSwish()
#
#         self.lstm1 = nn.LSTM(model_feature_num, 32, 2, dropout=drop, batch_first=True, bidirectional=True)  # 0为100-》50
#         self.lstm2 = nn.LSTM(64, 32, 2, dropout=drop, batch_first=True, bidirectional=True)
#         self.lstm3 = nn.LSTM(64, 32, 2, dropout=drop, batch_first=True, )
#         # self._feature_dim = 992+5952
#         self._feature_dim = feature_dim
#
#         self.out_features = feature_dim
#
#     def forward(self, xx):  # torch.Size([32, 6, 50])
#         # print(x.size())
#
#         covx = self.conv1(xx)  # torch.Size([32, 32, 31, 12])/1
#         covx = self._swish(covx)
#
#         covx = covx.view(covx.size(0), -1)  # torch.Size([32, 2688])
#
#         x = xx.squeeze(1)
#         x, _ = self.lstm1(x)  # torch.Size([32, 6, 64])
#         x, _ = self.lstm2(x)  # torch.Size([32, 6, 64])
#         x, _ = self.lstm3(x)  # torch.Size([32, 6, 32])
#         x = x.contiguous().view(x.size(0), -1)  # torch.Size([32, 192])
#         # print('x',x.size(),'covx',covx.size())
#         x = torch.cat([x, covx], 1)
#         # print('最后x',x.size(),'covx',covx.size()) #9*50 最后x torch.Size([2000, 7488]) covx torch.Size([2000, 7200])
#         print('dropout-backbones', 0.1)
#
#         return x
#
#     def output_num(self):
#         return self._feature_dim

# '''简单的backbones'''
# class Classification7(nn.Module):
#     def __init__(self, model_feature_num=50, model_time_num=9,
#                  feature_dim=800):  # model_feature_num=12,model_time_num=31，feature_dim=8160-最后的x的值
#         super(Classification7, self).__init__()
#         # print('type(model_time_num)',type(model_time_num)) #int
#         drop=0.3
#
#         self.conv1 = nn.Sequential(
#             get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
#                                                                                     kernel_size=(3,1), bias=False),
#             # ,padding='same' |kernel_size=3-》1
#             # nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=drop)
#         )
#         self.pool1 = nn.MaxPool2d((2, 1))
#
#         self.conv2 = nn.Sequential(
#             get_same_padding_conv2d(image_size=(model_time_num-int(2+1), model_feature_num))(in_channels=16, out_channels=8,
#                                                                                     kernel_size=(2,1), bias=False),
#             # ,padding='same' |kernel_size=3-》1
#             # nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=drop)
#         )
#         self.pool2 = nn.MaxPool2d((2, 1))
#
#
#         self._swish = MemoryEfficientSwish()
#
#         self._feature_dim = feature_dim
#
#         self.out_features = feature_dim
#
#     def forward(self, x):  # torch.Size([32, 6, 50])
#         # print(x.size())
#
#         x = self.conv1(x)  # torch.Size([32, 32, 31, 12])/1
#         x = self.pool1(x)
#         x = self.conv2(x)  # torch.Size([32, 32, 31, 12])/1
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         print('simplify scnn classification7')
#
#         return x
#
#     def output_num(self):
#         return self._feature_dim



class Classification8(nn.Module): #并联
    def __init__(self,model_feature_num=12,model_time_num=31,feature_dim=8160):
        super(Classification8, self).__init__()


        self.conv1 = nn.Sequential(
            get_same_padding_conv2d(image_size=(model_time_num,model_feature_num))(in_channels=1,out_channels=16, kernel_size=1, bias=False), # ,padding='same' |kernel_size=3-》1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self._swish = MemoryEfficientSwish()

        self.lstm1 = nn.LSTM(model_feature_num, 32, 2, batch_first=True, bidirectional=True) #0为100-》50
        self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True,)
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        dropout = 0.1
        bottleneck_list = [
        nn.Linear(self._feature_dim, 256),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Dropout(p=dropout),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list) #*用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222

        self.classifier_layer = nn.Linear(256, 2)
   
    def forward(self, xx): #torch.Size([32, 6, 50])
        # print(x.size())

        covx = self.conv1(xx) #torch.Size([32, 32, 31, 12])/1
        covx = self._swish(covx)

        covx = covx.view(covx.size(0), -1) #torch.Size([32, 2688])
        # print('covx.size()',covx.size()) #torch.Size([128, 7200])-39 //torch.Size([128, 134640])-2000
        # x = covx
        ####shuru
        ##conv  lstm
        ##res   res
        ##合并 特征合并
        ###fc


        ####shuru
        ##conv
        ##res
        ##lstm
        ##res
        ###fc


        ###A  点线面
        ##B
        ##C
        ##D 开发
        ###E 放开




        x = xx.squeeze(1)
        x,_ = self.lstm1(x) #torch.Size([32, 6, 64])
        x,_ = self.lstm2(x) #torch.Size([32, 6, 64])
        x,_ = self.lstm3(x) #torch.Size([32, 6, 32])
        x = x.contiguous().view(x.size(0), -1) #torch.Size([32, 192])
        # print(print('x.size()',x.size())) #x.size() torch.Size([128, 288])-39/2000
        #
        x = torch.cat([x,covx],1) #_feature_dim
        # print(print('x.size()', x.size()))#torch.Size([128, 7488])-39| torch.Size([128, 134928])-2000

        x = self.bottleneck_layer(x)
        x = self.classifier_layer(x)
        return x

    def output_num(self):
        # print('8-output_num')
        # print('self._feature_dim',self._feature_dim)
        return self._feature_dim


# class Classification9(nn.Module):  # 串联-跑直接分类预测的话
#     def __init__(self, model_feature_num=50, model_time_num=9, feature_dim=288):
#         super(Classification9, self).__init__()
#
#         dropout_conv = 0.1
#         self.conv1 = nn.Sequential(
#             get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
#                                                                                     kernel_size=2, bias=False),
#             # ,padding='same' |kernel_size=3-》1
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_conv)
#         )
#
#         self._swish = MemoryEfficientSwish()
#
#         dropout_lstm = 0.1
#         self.lstm1 = nn.LSTM(800 , 32, 2, batch_first=True, bidirectional=True,dropout = dropout_lstm)  # 0为100-》50 #note model_feature_num,第一个数为cnn的输出大小
#         self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True,dropout = dropout_lstm)
#         self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True,dropout = dropout_lstm)
#         # self._feature_dim = 992+5952
#         self._feature_dim = feature_dim
#
#         dropout = 0.1
#         bottleneck_list = [
#             nn.Linear(self._feature_dim, 256),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(256, 256),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#         ]
#         self.bottleneck_layer = nn.Sequential(*bottleneck_list)  # *用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222
#
#         self.classifier_layer = nn.Linear(256, 2)
#
#     def forward(self, xx):
#         # print('xx0.size()', xx.size())#torch.Size([128, 1, 9, 50])
#         # print('原始的数据',xx[0])
#         xx=torch.reshape(xx,(xx.size(0),9,-1,5))
#         # print('时间序列分块的数据',xx[0])
#         # print('xx1.size()', xx.size())#torch.Size([128, 9, 10, 5])
#         xx = xx.permute(1, 0, 2, 3)
#         # print('xx.size()',xx.size()) # torch.Size([9, 128, 10, 5])
#
#         xc=xx[0]
#         # print('输入的单机卷积数据',xc[0])
#         # print('xc.size()', xc.size())#torch.Size([128, 10, 5])
#         xc=torch.unsqueeze(xc, 1)
#         # print('xc.size()', xc.size())#torch.Size([128, 1, 10, 5])
#         covx = self.conv1(xc)
#         covx = self._swish(covx)
#         xl = covx.view(covx.size(0), -1)
#         # print('xl1.size',xl.size()) #torch.Size([128, 800])
#         for i in range(1,xx.size(0)):
#             x= xx[i]
#             x = torch.unsqueeze(x, 1)
#             # print('00x.size()',x.size()) #torch.Size([128, 1, 10, 5])
#             covx = self.conv1(x)
#             covx = self._swish(covx)
#             covx = covx.view(covx.size(0), -1)  # torch.Size([32, 2688])
#             xl = torch.cat([xl, covx], 1)  # _feature_dim
#             # print('xl.size', xl.size())#torch.Size([128, 1600(+800+...)-7200])
#
#         xl = xl.reshape(xl.size(0),9,-1)
#         # print('xl2.size', xl.size()) #xl2.size最后一个作为lstm的输入维数torch.Size #torch.Size([128, 9, 800])
#         x, _ = self.lstm1(xl)
#         # print('xlstm.size', x.size())#torch.Size([128, 9, 64])
#         x, _ = self.lstm2(x)  #
#         # print('xlstm.size', x.size())  #torch.Size([128, 9, 64])
#         x, _ = self.lstm3(x)  #
#         # print('xlstm.size', x.size())  #xlstm.size torch.Size([128, 9, 32])
#         x = x.contiguous().view(x.size(0), -1)  ##最后这个输出_feature_dim
#         # print('x.size',x.size()) #torch.Size([128, 288])
#
#         # _feature_dim
#
#         x = self.bottleneck_layer(x)
#         x = self.classifier_layer(x)
#         return x
#
#     def output_num(self):
#         # print('8-output_num')
#         # print('self._feature_dim',self._feature_dim)
#         return self._feature_dim


class Classification9(nn.Module):  # 串联-跑backbones的话
    def __init__(self, model_feature_num=50, model_time_num=9, feature_dim=288): #note feature_dim=320
        super(Classification9, self).__init__()

        dropout_conv = 0.3
        self.conv1 = nn.Sequential(
            get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
                                                                                    kernel_size=2, bias=False),
            # ,padding='same' |kernel_size=3-》1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_conv)
        )

        self._swish = MemoryEfficientSwish()

        dropout_lstm = 0.3
        self.lstm1 = nn.LSTM(800 , 32, 2, batch_first=True, bidirectional=True,dropout = dropout_lstm)  # 0为100-》50 #note model_feature_num,第一个数为cnn的输出大小,##39节点800，2000节点14960
        self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True,dropout = dropout_lstm)
        self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True,dropout = dropout_lstm)
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        dropout = 0.1
        bottleneck_list = [
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)  # *用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222

        self.classifier_layer = nn.Linear(256, 2)

    def forward(self, xx):
        # print('xx0.size()', xx.size())#torch.Size([128, 1, 9, 50])
        # print('原始的数据',xx.size(),xx[0])
        xx=torch.reshape(xx,(xx.size(0),9,-1,5)) #note xx=torch.reshape(xx,(xx.size(0),9,-1,5))
        # print('时间序列分块的数据',xx[0])
        # print('xx1.size()', xx.size())#torch.Size([128, 9, 10, 5])
        xx = xx.permute(1, 0, 2, 3)
        # print('xx.size()',xx.size()) # torch.Size([9, 128, 10, 5])

        xc=xx[0]
        # print('输入的单机卷积数据',xc[0])
        # print('xc.size()', xc.size())#torch.Size([128, 10, 5])
        xc=torch.unsqueeze(xc, 1)
        # print('xc.size()', xc.size())#torch.Size([128, 1, 10, 5])
        covx = self.conv1(xc)
        covx = self._swish(covx)
        xl = covx.view(covx.size(0), -1)
        # print('xl1.size',xl.size()) #torch.Size([128, 800])
        for i in range(1,xx.size(0)):
            x= xx[i]
            x = torch.unsqueeze(x, 1)
            # print('00x.size()',x.size()) #torch.Size([128, 1, 10, 5])
            covx = self.conv1(x)
            covx = self._swish(covx)
            covx = covx.view(covx.size(0), -1)  # torch.Size([32, 2688])
            xl = torch.cat([xl, covx], 1)  # _feature_dim
            # print('xl.size', xl.size())#torch.Size([128, 1600(+800+...)-7200])

        xl = xl.reshape(xl.size(0),9,-1) #note 10
        # print('xl2.size', xl.size()) #xl2.size最后一个作为lstm的输入维数torch.Size #torch.Size([128, 9, 800])
        x, _ = self.lstm1(xl)
        # print('xlstm.size', x.size())#torch.Size([128, 9, 64])
        x, _ = self.lstm2(x)  #
        # print('xlstm.size', x.size())  #torch.Size([128, 9, 64])
        x, _ = self.lstm3(x)  #
        # print('xlstm.size', x.size())  #xlstm.size torch.Size([128, 9, 32])
        x = x.contiguous().view(x.size(0), -1)  ##最后这个输出_feature_dim 都是288
        # print('x.size', x.size())
        return x

    def output_num(self):
        # print('8-output_num')
        # print('self._feature_dim',self._feature_dim)
        return self._feature_dim

class Classification10(nn.Module):  # 串联-跑直接分类预测的话
    def __init__(self, model_feature_num=50, model_time_num=9, feature_dim=288):
        super(Classification10, self).__init__()

        dropout_conv = 0.1
        self.conv1 = nn.Sequential(
            get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
                                                                                    kernel_size=2, bias=False),
            # ,padding='same' |kernel_size=3-》1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_conv)
        )

        self._swish = MemoryEfficientSwish()

        dropout_lstm = 0.1
        self.lstm1 = nn.LSTM(800 , 32, 2, batch_first=True, bidirectional=True,dropout = dropout_lstm)  # 0为100-》50 #note model_feature_num,第一个数为cnn的输出大小
        self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True,dropout = dropout_lstm)
        self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True,dropout = dropout_lstm)
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        dropout = 0.1
        bottleneck_list = [
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)  # *用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222

        self.classifier_layer = nn.Linear(256, 2)

    def forward(self, xx):
        # print('xx0.size()', xx.size())#torch.Size([128, 1, 9, 50])
        # print('原始的数据',xx[0])
        xx=torch.reshape(xx,(xx.size(0),9,-1,5))
        # print('时间序列分块的数据',xx[0])
        # print('xx1.size()', xx.size())#torch.Size([128, 9, 10, 5])
        xx = xx.permute(1, 0, 2, 3)
        # print('xx.size()',xx.size()) # torch.Size([9, 128, 10, 5])

        xc=xx[0]
        # print('输入的单机卷积数据',xc[0])
        # print('xc.size()', xc.size())#torch.Size([128, 10, 5])
        xc=torch.unsqueeze(xc, 1)
        # print('xc.size()', xc.size())#torch.Size([128, 1, 10, 5])
        covx = self.conv1(xc)
        covx = self._swish(covx)
        xl = covx.view(covx.size(0), -1)
        # print('xl1.size',xl.size()) #torch.Size([128, 800])
        for i in range(1,xx.size(0)):
            x= xx[i]
            x = torch.unsqueeze(x, 1)
            # print('00x.size()',x.size()) #torch.Size([128, 1, 10, 5])
            covx = self.conv1(x)
            covx = self._swish(covx)
            covx = covx.view(covx.size(0), -1)  # torch.Size([32, 2688])
            xl = torch.cat([xl, covx], 1)  # _feature_dim
            # print('xl.size', xl.size())#torch.Size([128, 1600(+800+...)-7200])

        xl = xl.reshape(xl.size(0),9,-1)
        # print('xl2.size', xl.size()) #xl2.size最后一个作为lstm的输入维数torch.Size #torch.Size([128, 9, 800])
        x, _ = self.lstm1(xl)
        # print('xlstm.size', x.size())#torch.Size([128, 9, 64])
        x, _ = self.lstm2(x)  #
        # print('xlstm.size', x.size())  #torch.Size([128, 9, 64])
        x, _ = self.lstm3(x)  #
        # print('xlstm.size', x.size())  #xlstm.size torch.Size([128, 9, 32])
        x = x.contiguous().view(x.size(0), -1)  ##最后这个输出_feature_dim

        x = self.bottleneck_layer(x)
        x = self.classifier_layer(x)

        return x

    def output_num(self):
        # print('8-output_num')
        # print('self._feature_dim',self._feature_dim)
        return self._feature_dim


#
# class Classification9(nn.Module):  # 串联
#     def __init__(self, model_feature_num=12, model_time_num=31, feature_dim=8160):
#         super(Classification8, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
#                                                                                     kernel_size=1, bias=False),
#             # ,padding='same' |kernel_size=3-》1
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True)
#         )
#
#         self._swish = MemoryEfficientSwish()
#
#         self.lstm1 = nn.LSTM(model_feature_num, 32, 2, batch_first=True, bidirectional=True)  # 0为100-》50
#         self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True)
#         self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True, )
#         # self._feature_dim = 992+5952
#         self._feature_dim = feature_dim
#
#         dropout = 0.1
#         bottleneck_list = [
#             nn.Linear(self._feature_dim, 256),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(256, 256),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#         ]
#         self.bottleneck_layer = nn.Sequential(*bottleneck_list)  # *用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222
#
#         self.classifier_layer = nn.Linear(256, 2)
#
#     def forward(self, xx):  # torch.Size([32, 6, 50])
#         # print(x.size())
#
#         covx = self.conv1(xx)  # torch.Size([32, 32, 31, 12])/1
#         covx = self._swish(covx)
#
#         covx = covx.view(covx.size(0), -1)  # torch.Size([32, 2688])
#         ####shuru
#         ##conv  lstm
#         ##res   res
#         ##合并 特征合并
#         ###fc
#
#         ####shuru
#         ##conv
#         ##res
#         ##lstm
#         ##res
#         ###fc
#
#         ###A  点线面
#         ##B
#         ##C
#         ##D 开发
#         ###E 放开
#
#         x = xx.squeeze(1)
#         x, _ = self.lstm1(x)  # torch.Size([32, 6, 64])
#         x, _ = self.lstm2(x)  # torch.Size([32, 6, 64])
#         x, _ = self.lstm3(x)  # torch.Size([32, 6, 32])
#         x = x.contiguous().view(x.size(0), -1)  # torch.Size([32, 192])
#
#         x = torch.cat([x, covx], 1)  # _feature_dim
#
#         x = self.bottleneck_layer(x)
#         x = self.classifier_layer(x)
#         return x
#
#     def output_num(self):
#         # print('8-output_num')
#         # print('self._feature_dim',self._feature_dim)
#         return self._feature_dim


class SVM(nn.Module):
    def __init__(self, model_feature_num=12, model_time_num=31, feature_dim=8160):
        super(SVM, self).__init__()
        self.layer = nn.Linear(8415, 2)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        # print('x.size()',x.size())#torch.Size([128, 450])
        x = self.layer(x)
        return x


class CNN(nn.Module):
    def __init__(self, model_feature_num=12, model_time_num=31, feature_dim=8160):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            get_same_padding_conv2d(image_size=(model_time_num, model_feature_num))(in_channels=1, out_channels=16,
                                                                                    kernel_size=3, bias=False),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d((2, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1),  # ,padding='same' | kernel_size=3-》2
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d((2, 1))

        self._swish = MemoryEfficientSwish()
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        dropout = 0.1
        bottleneck_list = [
            nn.Linear(self._feature_dim, 64),
            nn.ReLU(True),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)  # *用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222

        self.classifier_layer = nn.Linear(64, 2)

    def forward(self, xx):  # torch.Size([32, 6, 50])
        # print(x.size())

        covx = self.conv1(xx)  # torch.Size([32, 32, 31, 12])/1
        covx = self._swish(covx)

        covx = covx.view(covx.size(0), -1)  # torch.Size([32, 2688])-torch.Size([128, 7200])

        x = covx  # _feature_dim
        # print('x.size()',x.size())# x.size() torch.Size([128, 7200])

        x = self.bottleneck_layer(x)
        x = self.classifier_layer(x)
        return x

    def output_num(self):
        # print('8-output_num')
        # print('self._feature_dim',self._feature_dim)
        return self._feature_dim

class LSTM(nn.Module):
    def __init__(self, model_feature_num=12, model_time_num=31, feature_dim=8160):
        super(LSTM, self).__init__()

        self.lstm1 = nn.LSTM(model_feature_num, 32, 2, batch_first=True, bidirectional=True)  # 0为100-》50
        self.lstm2 = nn.LSTM(64, 32, 2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(64, 32, 2, batch_first=True, )
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        self._swish = MemoryEfficientSwish()
        # self._feature_dim = 992+5952
        self._feature_dim = feature_dim

        dropout = 0.1
        bottleneck_list = [
            nn.Linear(self._feature_dim, 64),
            nn.ReLU(True),
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)  # *用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222

        self.classifier_layer = nn.Linear(64, 2)

    def forward(self, xx):  # torch.Size([32, 6, 50])
        x = xx.squeeze(1)
        x, _ = self.lstm1(x)  # torch.Size([32, 6, 64])
        x, _ = self.lstm2(x)  # torch.Size([32, 6, 64])
        x, _ = self.lstm3(x)  # torch.Size([32, 6, 32])
        x = x.contiguous().view(x.size(0), -1)  # torch.Size([32, 192])
        # print('x.size()',x.size()) #x.size() torch.Size([128, 288])

        x = self.bottleneck_layer(x)
        x = self.classifier_layer(x)
        return x

    def output_num(self):
        # print('8-output_num')
        # print('self._feature_dim',self._feature_dim)
        return self._feature_dim



if __name__=='__main__':
    # import numpy as np
    # n1 = np.empty([1, 31, 12])
    # print(n1.shape)
    # n2 = np.concatenate((n1,n1,n1[:,:,:7]),axis=2)
    # print(n2.shape)
    # exec()

    # Conv2d = get_same_padding_conv2d(image_size=(31,12))

    # model = Classification3()
    # input_v = torch.randn((1,6,100)).unsqueeze(0)
    # model = Classification7(model_feature_num=15,model_time_num=30)
    # input_v = torch.randn((1,30,15)).unsqueeze(0)

    # model = Classification7(model_feature_num=45,model_time_num=50)
    # input_v = torch.randn((1,50,45)).unsqueeze(0)

    # model = Classification7(model_feature_num=45,model_time_num=30)
    # input_v = torch.randn((1,30,45)).unsqueeze(0)

    # model = Classification7(model_feature_num=50,model_time_num=45,feature_dim=37440)
    # input_v = torch.randn((1,45,50)).unsqueeze(0)

    model = Classification8(model_feature_num=50,model_time_num=9,feature_dim=7488) # model_time_num=45->>9 , feature_dim=37440 -->7488

    print(model)

    # for param in model.named_parameters():
    #     if 'bottleneck_layer' in param[0]:
    #         require_grad_bool =True
    #     if require_grad_bool:
            
    #         param.require_grad = True
    #     print(param[0],param[1].require_grad)
    require_grad_bool = False
    for param in model.named_parameters():
        param[1].requires_grad = False
        if 'bottleneck_layer' in param[0]:
            require_grad_bool =True
        if require_grad_bool:
            param[1].requires_grad = True
        print(param[0],param[1].requires_grad)
        # exec()

    input_v = torch.randn((1,9,50)).unsqueeze(0) #(1,45,50)->>(1,9,50)



    # model = Classification8(model_feature_num=30,model_time_num=45,feature_dim=23040)
    # input_v = torch.randn((1,45,30)).unsqueeze(0)  
    # print(model)
    device = torch.device('cpu')
    model.eval()
    # input_v = torch.randn((30,12)).unsqueeze(0)
    # input_v = torch.randn((1,31,12)).unsqueeze(0)

    # input_v = torch.randn((31,12)).unsqueeze(0)
    # input_v = torch.randn((3,31,31)).unsqueeze(0)
    # print(input_v)
# torch.Size([2000, 1, 31, 12])
# torch.Size([2000])

    model.to(device)
    input_v = input_v.to(device)
    import time

    with torch.no_grad():
        for i in [1]:
            start_time = time.time()
            x = model(input_v)
            print(x.size())
