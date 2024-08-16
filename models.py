import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
import numpy as np

class TransferNet(nn.Module):
    def __init__(self,args, num_class=2, base_net='classification9', transfer_loss='daan',
    use_bottleneck=False, bottleneck_width=256, max_iter=1000,source_weights_train=[1,1], **kwargs): #'classification3->'classification7' #num_class->num_class=2,base_net:resnet50->classification3
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net,args)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            # print('self.base_network',self.base_network)
            # #定义全连接层
            # bottleneck_list = [
            #     nn.Linear(self.base_network.output_num(), bottleneck_width),
            #     nn.ReLU(),
            # ]
            # self.bottleneck_layer = nn.Sequential(*bottleneck_list) #*用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222

            # dropout = 0.5#####1110 1111 
            dropout = 0.5
            bottleneck_list = [
            nn.Linear(self.base_network.output_num(), 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list) #*用于迭代取出list中内容，如x = [1,2,2,2,2,2] print(*x)-》122222


            feature_dim = bottleneck_width
        else:
            # print('self.base_network',self.base_network)
            # print('self.base_network.output_num()',self.base_network.output_num())
            feature_dim = self.base_network.output_num()

        
        self.classifier_layer = nn.Linear(feature_dim, num_class)

        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)

        self.source_weight=torch.from_numpy(np.array(source_weights_train)).float()
        # self.criterion = torch.nn.CrossEntropyLoss(weight=)


    def forward(self, source, target, source_label,is_train=True):

        if is_train:
            source = self.base_network(source)
            target = self.base_network(target)
            # print("model-source.size()",source.size())
            # print("model-target.size()", target.size())
            if self.use_bottleneck:
                # print('000000000')
                source = self.bottleneck_layer(source)
                target = self.bottleneck_layer(target)
                # print("111111")
            # classification
            # print('self.classifier_layer',self.classifier_layer)
            source_clf = self.classifier_layer(source)
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.source_weight.to(source.device))
            clf_loss = self.criterion(source_clf, source_label.long())
            # transfer
            kwargs = {}
            if self.transfer_loss == "lmmd":
                kwargs['source_label'] = source_label
                target_clf = self.classifier_layer(target)
                kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            elif self.transfer_loss == "daan":
                # print('使用DAAN损失函数') --使用了
                source_clf = self.classifier_layer(source)
                kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
                target_clf = self.classifier_layer(target)
                kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            elif self.transfer_loss == 'bnm':
                tar_clf = self.classifier_layer(target)
                target = nn.Softmax(dim=1)(tar_clf)
            # print('source.item(), target.item()', source, target)
            transfer_loss, globalloss, localloss, w = self.adapt_loss(source, target, **kwargs)
            # print('transfer_loss.item()',transfer_loss.item())
            return clf_loss, transfer_loss,source_clf, globalloss, localloss, w

        else:
            features = self.base_network(target)
            if self.use_bottleneck:
                features = self.bottleneck_layer(features)
            clf = self.classifier_layer(features)
            return clf
    # def get_parameters(self, initial_lr=1.0):
    #     params = [
    #         {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
    #         {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
    #     ]
    #     if self.use_bottleneck:
    #         params.append(
    #             {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
    #         )
    #     # Loss-dependent
    #     if self.transfer_loss == "adv":
    #         params.append(
    #             {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
    #         )
    #     elif self.transfer_loss == "daan":
    #         params.append(
    #             {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
    #         )
    #         params.append(
    #             {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
    #         )
    #     return params

    def predict(self, x):
        features = self.base_network(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        clf = self.classifier_layer(features)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

    def basenet_TSNE(self, x):
        features = self.base_network(x)
        return features
          