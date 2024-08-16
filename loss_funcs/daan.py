from loss_funcs.adv import *

class DAANLoss(AdversarialLoss, LambdaSheduler):
    def __init__(self, num_class, gamma=1.0, max_iter=1000, **kwargs):
        super(DAANLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        self.num_class = num_class
        ###子域判别器，加入条件对抗判别器
        self.local_classifiers = torch.nn.ModuleList()
        for _ in range(num_class):
            self.local_classifiers.append(Discriminator())

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5
        # self.dynamic_factor = 1
        self.d_l0 , self.d_l1 =0, 0

    ### 动态调节因子dynamic_factor
    ###需要源域和目标域经过分类器预测的source_logits和target_logits作为BCELOSS的输入求出源域的全局损失和local 损失，目标域也是如此。最后计算得出源域和目标域的全局损失和local损失。
    def forward(self, source, target, source_logits, target_logits):
        lamb = self.lamb()
        self.step()
        source_loss_g = self.get_adversarial_result(source, True, lamb)
        target_loss_g = self.get_adversarial_result(target, False, lamb)
        source_loss_l,source_loss_l0,source_loss_l1 = self.get_local_adversarial_result(source, source_logits, True, lamb)
        target_loss_l,target_loss_l0,target_loss_l1 = self.get_local_adversarial_result(target, target_logits, False, lamb)
        # global_loss = 0.5 * (source_loss_g + target_loss_g) * 0.05
        # local_loss = 0.5 * (source_loss_l + target_loss_l) * 0.01 #得到的值是所有类的总和，最后要除类总数，得到一个子类loss的平均值
        # # 源域和目标域比例是1：1，所以总数×0.5

        global_loss = source_loss_g + target_loss_g
        local_loss = source_loss_l + target_loss_l # 得到的值是所有类的总和，最后要除类总数，得到一个子类loss的平均值
        # print('source_loss_g,target_loss_g,source_loss_l,target_loss_l,global_loss-s+t,local_loss-s+t,local_loss-s+t0,local_loss-s+t','\n',
        #       source_loss_g.cpu().item(),target_loss_g.cpu().item(),source_loss_l.cpu().item(),target_loss_l.cpu().item(),
        #       (source_loss_g + target_loss_g).cpu().item(),(source_loss_l + target_loss_l).cpu().item(),
        #       (source_loss_l0 + target_loss_l0).cpu().item(),(source_loss_l1 + target_loss_l1).cpu().item())

        # global_loss_ = source_loss_g + target_loss_g
        # local_loss_ = source_loss_l + target_loss_l
        local_loss_0 = source_loss_l0 + target_loss_l0
        local_loss_1 = source_loss_l1 + target_loss_l1
        # d_g = 2 * (1 - 2 * global_loss_.cpu().item())
        # d_l =  2 * (1 - 2 * (local_loss_ / self.num_class).cpu().item())
        # d_l0 = 2 * (1 - 2 * (local_loss_0).cpu().item())
        # d_l1 = 2 * (1 - 2 * (local_loss_1).cpu().item())
        # w=d_g/(d_g+d_l)
        # print('d_g,d_l,d_l0,d_l1,w',d_g,d_l,d_l0,d_l1,w)
        # print('global_loss_,local_loss_,local_loss_0,local_loss_1',global_loss_,local_loss_,local_loss_0,local_loss_1)



        self.d_g = self.d_g + 2 * (1 - 2 * global_loss.cpu().item()) #定义全局A距离
        self.d_l = self.d_l + 2 * (1 - 2 * (local_loss / self.num_class).cpu().item())

        self.d_l0 = self.d_l + 2 * (1 - 2 * (local_loss_0).cpu().item())
        self.d_l1 = self.d_l + 2 * (1 - 2 * (local_loss_1).cpu().item())
        # print('d_g,d_l,d_l0,d_l1',self.d_g,self.d_l,self.d_l0,self.d_l1)
        # self.update_dynamic_factor(epoch_length=111)
        # print('self.d_g,self.d_l,self.dynamic_factor,dynamic_factor,global_loss,local_loss',self.d_g,self.d_l,self.dynamic_factor,global_loss,local_loss)

        # err(h)表示的是平均值

        adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss#源域和目标域判断的损失，域对抗discrimination的总损失
        globalloss = global_loss
        localloss = local_loss
        w = self.dynamic_factor

        """
         adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss
          self.dynamic_factor 
        """
        return adv_loss , globalloss, localloss, w
        ### adv_loss 是新的域判别器损失
    
    def get_local_adversarial_result(self, x, logits, c, source=True, lamb=1.0):
        loss_fn = nn.BCELoss()
        x = ReverseLayerF.apply(x, lamb) #这个x是输入类判别器中的特征值
        loss_adv = 0.0
        loss_adv0 =0.0
        loss_adv1 =0.0

        for c in range(self.num_class):
            logits_c = logits[:, c].reshape((logits.shape[0],1)) # (B, 1)
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            # 第几个分类器，使用这一类的特征数据
            device = domain_pred.device
            if source:
                domain_label = torch.ones(len(x), 1).long()
            else:
                domain_label = torch.zeros(len(x), 1).long()
            if c==0:
                loss_adv0 = loss_adv0 + loss_fn(domain_pred, domain_label.float().to(device))
            if c==1:
                loss_adv1 = loss_adv1 + loss_fn(domain_pred, domain_label.float().to(device))

            loss_adv = loss_adv + loss_fn(domain_pred, domain_label.float().to(device))
            # print('计算类别的损失，类别',c,'类别损失总数',loss_adv)
        # print('类别损失：总、0、1：',loss_adv.cpu().item(), loss_adv0.cpu().item(), loss_adv1.cpu().item())
        return loss_adv,loss_adv0,loss_adv1

    # 更新动态调节因子w的函数
    def update_dynamic_factor(self, epoch_length):
        print('运行updata,epochlenth',epoch_length)
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
            # self.dynamic_factor = 1
            print('初始条件:self.d_g,self.d_l,self.dynamic_factor',self.d_g,self.d_l,self.dynamic_factor )
            """
            初始化参数的条件
            """
        else:
            print('epoch_length,self.d_g,self.d_l(没有除)', epoch_length, self.d_g, self.d_l)
            self.d_g = self.d_g / epoch_length ##平均值
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l) 
            print('自适应修改:self.d_g,self.d_l,self.dynamic_factor ',self.d_g,self.d_l,self.dynamic_factor )
        self.d_g, self.d_l = 0, 0