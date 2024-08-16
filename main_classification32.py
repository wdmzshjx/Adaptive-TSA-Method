import os
import argparse
import torch
import distributed_utils_classification as distributed_utils
import numpy as np
import torch.backends.cudnn as cudnn
import time
from distributed_utils_classification import is_main_process,MetricLogger
from typing import Iterable, Optional
import torch.optim as optim
import datetime
from optim_factory import create_optimizer
import  pickle



from dataset_classification import dataset_classi,DatasetGenerate
from torch.autograd import Variable
from torch.utils.data import DataLoader

from backbones import Classification8,Classification9,Classification10
from backbones import SVM,CNN,LSTM
import logging
from torch import nn
from dataset_classification import dataset_classi_new


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script for image matting', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int) #100->150;300->100
    parser.add_argument('--update_freq', default=1, type=int, #1相当于每个batch反向传播
                        help='gradient accumulation steps')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', #重加载、没用到
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # parser.add_argument('--log_dir', default='./distributed_log/main_classification_source' , #'./distributed_log/base_line2' ->>'./distributed_log/main_classification_source'
    #                     help='path where to tensorboard log')
    parser.add_argument('--log_dir', default='.\LOG\\3931_1',#'E:\CODE\CODE2\\LOG\\3914_0\main_classification_target' | '.\LOG\\20002_13\main_classification_target' 'E:\CODE\CODE12\PKL\\20002_13'
                        # './distributed_log/base_line2' ->>'./distributed_log/main_classification_source'
                        help='path where to tensorboard log')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")  

    #######warm up
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.00001')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    # parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--logname', type=str, default='train_log', help="name of the logging file")
    parser.add_argument('--save_ckpt_num', default=300, type=int)
    # parser.add_argument('--finetune', type=str, default='.//3914saved_models/u2net/u2net.pth', help="name of the logging file") #change 为了避免重复./saved_models/u2net/u2net.pth
    parser.add_argument('--model_name', type=str, default='eff3', help="model_name")
    # parser.add_argument('--class_names', type=list, default=['No','Yes'], help="class_names")

    parser.add_argument('--data_pkl_addr', type=str, default='E:\CODE\CODE2\PKL\\3931_1')  # 存训练输出数据的位置 #D盘满了，存到E盘
    parser.add_argument('--class_names',type=str, default='No,Yes', help="class_names")
    parser.add_argument('--domain_pkl', type=str, default='3931INDEX-0-SC_QD_RGB.pkl') #BASE4INDEX-0-SC_QD_RGB.pkl 3931INDEX-0-SC_QD_RGB.pkl
    parser.add_argument('--domain_new_pkl', type=str, default='BIANJIERC2Q20001INDEX-0-SC.pkl')
    parser.add_argument('--input_x_key', type=str, default='DATA0SC') # 'DATA0SC', 'DATA0QD', 'DATA0RGB' #SC'X'
    parser.add_argument('--input_y_key', type=str, default='WEN01INDEX') #'WEN01' #SC'y'
    parser.add_argument('--feature_dim', type=int, default=288) #串联特征提取器 39节点288 -->>37440| CL7488->SVM=450->CNN7200->LSTM288;39-7488 /2000yuan-134928、cnn-134640,LSTM288,SVM8415/2000mubiao-132048,cnn-131760,lstm-288
    return parser

def intersect_dicts(da, db, exclude=()): #名字维度相同的加载进来，其他的按模型来 load model
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def load_model(args): 

    # model = Classification8(model_feature_num=50,model_time_num=9,feature_dim=7488)
    # model = Classification8(model_feature_num=30,model_time_num=45,feature_dim=23040)

    model = Classification10(model_feature_num=50,model_time_num=9,feature_dim=args.feature_dim)
    # model = SVM(model_feature_num=935, model_time_num=9, feature_dim=args.feature_dim)
    # model = CNN(model_feature_num=50, model_time_num=9, feature_dim=args.feature_dim)
    # model = LSTM(model_feature_num=50, model_time_num=9, feature_dim=args.feature_dim)

    # #####note 迁移时加载模型 冻结骨干网络 note微调的时候用此，正常训练注掉
    # csd = torch.load('./LOG//20001_2/main_classification_target/checkpoint-best.pth', map_location='cpu') #checkpoint-93.pth
    # # csd = torch.load('E:\CODE\CODE12/LOG//20002_11/main_classification_target/checkpoint-2.pth',map_location='cpu')  # checkpoint-93.pth
    # model.load_state_dict(csd['model'], strict=True)  # load #将csd中保存的网络参数值赋值给现在的网络

    '''以下为load模型时使用'''
    # csd = torch.load('./LOG//3931_7/main_classification_target/checkpoint-4.pth',map_location='cpu')  # checkpoint-93.pth
    # # # csd = intersect_dicts(csd, model.state_dict(), exclude=[])  # intersect
    # csd_model = csd['model']
    # csd_keys = csd['model'].keys()
    params = {}
    # gudingload = ['conv1','lstm1', 'lstm2','lstm3'] #'conv1','lstm1','lstm2','lstm3','bottleneck_layer','classifier_layer'
    # # # for key0 in csd_keys:
    # # #     if key0.split('.')[0] in gudingload:
    # # #         print('key0', key0)
    # # #         params[key0] = csd_model[key0]
    # # # model.load_state_dict(params, strict=False)  # note False 仅load固定层的值
    # # # #
    # for param in model.named_parameters():
    #     if param[0].split('.')[0] in gudingload:
    #         print('固定', param[0].split('.')[0])
    #         param[1].requires_grad = False
    #     print(param[0], param[1].requires_grad)
    # '''仅重新训练retrainload中层，其他固定'''
    # retrainload = ['bottleneck_layer','classifier_layer']  # 'conv1','lstm1','lstm2','lstm3','bottleneck_layer','classifier_layer'
    # for param in model.named_parameters():
    #     param[1].requires_grad = False
    #     if param[0].split('.')[0] in retrainload:
    #         print('固定', param[0].split('.')[0])
    #         param[1].requires_grad = True
    #     print(param[0], param[1].requires_grad)
    #
    #
    # model.load_state_dict(csd['model'], strict=True)  # load所有值 #将csd中保存的网络参数值赋值给现在的网络
    start_epoch = 0
    return model, start_epoch

def load_dataset(args):
    # sample_pkl = 'data_xywustime_nh9.pkl'
    # sample_pkl = args.domain_pkl #note 不增加边界样本时
    sample_pkl = args.domain_pkl
    sample_new_pkl = args.domain_new_pkl
    input_x_key=args.input_x_key
    input_y_key=args.input_y_key
    x_train, y_train, x_test, y_test, x_ver, y_ver, x_test_all, y_test_all, weights_train = dataset_classi(sample_pkl,input_x_key=input_x_key,input_y_key=input_y_key)  # note 不增加边界样本时

    # x_train,y_train,x_test,y_test,x_ver,y_ver,x_test_all,y_test_all,weights_train = dataset_classi_new(sample_pkl,sample_new_pkl,input_x_key=input_x_key,input_y_key=input_y_key) #note 增加边界样本时

    args.weights_train = weights_train
    # print('train_len:  ',len(x_train)," ",'valid_len:  ',len(valid_data))
    dataset_train = DatasetGenerate(x_train,y_train,data_type='train')
    dataset_val = DatasetGenerate(x_test,y_test,data_type='val') #note用的test来做验证数据
    # return dataset_val,dataset_train
    return dataset_train,dataset_val

def tensor_print(res):
    for t in res:
        print(torch.max(t),torch.min(t))

def train(args, model,data_loader: Iterable,optimizer,device: torch.device, epoch, 
    log_writer=None,start_steps=None,num_training_steps_per_epoch=None, update_freq=None,lr_schedule_values=None,sampler_train=None):

    metric_logger = MetricLogger(delimiter="  ")
    model.train(True)
    print_freq = max(1,int(num_training_steps_per_epoch/2)) 

    header = 'Epoch Train: [{}]'.format(epoch)
    optimizer.zero_grad()
    sampler_train.set_epoch(epoch+1)

    train_pout = []
    train_py=[]
    train_zq =[]
    train_labelindex = []


    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] 
                    # * param_group["lr_scale"]


        batch_new = []
        for item in batch:
            item = Variable(item).cuda(device, non_blocking=True)
            batch_new.append(item)
        # [inputs_v,labels_v] = batch_new #note list时的label
        """加label-index开始"""
        [inputs_v, labelsindex_v] = batch_new
        # print('labelsindex_v',labelsindex_v)
        labels_v = labelsindex_v[:,0]
        # print(labels_v)
        # labels_v = int(labels_v)
        """加label-index结束"""

        # image_tensor,mask_tensor,edge_tensor,body_tensor
        # forward + backward + optimize
        output= model(inputs_v)
        # print('inputs_v',inputs_v.shape,inputs_v[0])
        # print('labels_v',labels_v.shape,labels_v)
        # print('output',output.shape,output)
        weight=torch.from_numpy(np.array(args.weights_train)).float()
        # print('weight',weight)#weight tensor([1.9355, 1.0000])
        criterion = nn.CrossEntropyLoss(weight=weight.to(device))
        # loss = criterion(output, labels_v) #note list
        loss = criterion(output, labels_v.long())  #


            
        loss_value = loss.item()

        loss /= update_freq

        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        # print('output.max(-1)', output.max(-1)) #每一个样本（每一行）概率的最大值
        # print('output.max(-1)[-1]',output.max(-1)[-1]) #note这里是最后的分类结果 每一个样本（每一行）概率的最大值 ,最大值的索引，即返回0或者1，正好与我们原始数据的label相对应
        # print('type(output.max(-1)[-1] == labels_v', type(output.max(-1)[-1] == labels_v))
        # print('output.max(-1)[-1] == labels_v',output.max(-1)[-1] == labels_v) #对的true，错的false
        # print('(output.max(-1)[-1] == labels_v).float()',(output.max(-1)[-1] == labels_v).float()) #true-》1，false——》0
        # print('(output.max(-1)[-1] == labels_v).float().mean()',(output.max(-1)[-1] == labels_v).float().mean()) #取0、1的均值，代表正确率
        class_acc = (output.max(-1)[-1] == labels_v).float().mean() #.max(0) #在第一个维度求最大值#output[batchsize行，2分类列] .max(-1)取每行最大值,获得values=tensor([ ])，索引值indices=tensor([2, 2])) #[-1]表示获得索引值，即二分类结果
        metric_logger.update(class_acc=class_acc)

        train_pout.append(output)
        train_py.append(output.max(-1)[-1])
        train_zq.append((output.max(-1)[-1] == labels_v).float())
        train_labelindex.append(labelsindex_v)

        # nn.utils.clip_grad_norm_(model.parameters(), 2)

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        # del temporary outputs and loss
        del inputs_v,labels_v,loss


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        args.logging.info("Train GFM-Epoch[{}/{}] Lr:{:.8f} Loss:{:.5f}".format(
                epoch, args.epochs,optimizer.param_groups[0]['lr']
                , stats['loss']))

        for key in stats.keys():
            log_writer.update_epoch(**{key:stats[key]}, head="epoch_train")

        # log_writer.set_step_epoch()
    return stats ,train_pout, train_py, train_zq,train_labelindex

def eval(args, model,data_loader: Iterable,optimizer,device: torch.device, epoch, log_writer=None,start_steps=None,num_training_steps_per_epoch=None, update_freq=None):

    metric_logger = MetricLogger(delimiter="  ")
    model.eval()
    print_freq = max(1,int(num_training_steps_per_epoch/2)) 
    header = 'Epoch Val: [{}]'.format(epoch)

    eval_pout = []
    eval_py = []
    eval_zq = []
    eval_labelindex = []

    with torch.no_grad():
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration

            batch_new = []
            for item in batch:
                item = Variable(item).cuda(device, non_blocking=True)
                batch_new.append(item)
            # [inputs_v,labels_v] = batch_new #note list时的label
            """加label-index开始"""
            [inputs_v, labelsindex_v] = batch_new
            # print('labelsindex_v',labelsindex_v)
            labels_v = labelsindex_v[:, 0]
            """加label-index结束"""

            # image_tensor,mask_tensor,edge_tensor,body_tensor
            # forward + backward + optimize
            output= model(inputs_v)
            weight=torch.from_numpy(np.array(args.weights_train)).float() #只有在算loss的时候算权重
            criterion = nn.CrossEntropyLoss(weight=weight.to(device))
            # loss = criterion(output, labels_v) #note list
            loss = criterion(output, labels_v.long())  # label-index

            loss_value = loss.item()

            class_acc = (output.max(-1)[-1] == labels_v).float().mean()
            metric_logger.update(class_acc=class_acc)

            eval_pout.append(output)
            eval_py.append(output.max(-1)[-1])
            eval_zq.append((output.max(-1)[-1] == labels_v).float())
            eval_labelindex.append(labelsindex_v)

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)

            del inputs_v,labels_v,loss

    # gather the stats from all processes #做数据同步，把一个batch所有的数算出来之后一起算
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        args.logging.info("Val GFM-Epoch[{}/{}] Lr:{:.8f} Loss:{:.5f}".format(
                epoch, args.epochs,optimizer.param_groups[0]['lr']
                , stats['loss']))

        for key in stats.keys():
            log_writer.update_epoch(**{key:stats[key]}, head="epoch_val")
  
    return stats ,eval_pout, eval_py, eval_zq,eval_labelindex

def main(args):
    distributed_utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    ####np.random.seed(seed)#####数据增强需要 需要注释掉

    cuda_deterministic = False
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    num_tasks = distributed_utils.get_world_size()
    global_rank = distributed_utils.get_rank()


    #####保存event 日志
    if global_rank == 0 and args.log_dir is not None:

        os.makedirs(args.log_dir, exist_ok=True)
        now = datetime.datetime.now()
        logging_filename =args.log_dir+"/" +args.logname+'_'+now.strftime("%Y-%m-%d-%H:%M")+'.log'
        print('logging_filename',logging_filename)
        print('logging.INFO',logging.INFO)
        logging.basicConfig(filename=logging_filename, level=logging.INFO)
        args.logging = logging
        log_writer = distributed_utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    #####################加载自己的数据
    dataset_train,dataset_val = load_dataset(args)
    #####分布式训练 与验证
    sampler_train = torch.utils.data.DistributedSampler(  #torch.utils.data.DistributedSampler 不使用则数据都在一个卡上训练，如果使用则数据分到多个卡上
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    data_loader_train = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset_train, sampler=sampler_train,#自定义从数据集中采样的策略，如果制定了采样策略，shuffle则必须为False
        batch_size=args.batch_size,# mini batch size
        num_workers=args.num_workers, # 多线程来读数据
        pin_memory=args.pin_mem, #当计算机的内存充足的时候，可以设置pin_memory=True。当系统卡主，或者内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确定每一个炼丹玩家都有高端设备，因此pin_memory默认设置为False.
        drop_last=True,
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    ####创建模型
    model, start_epoch = load_model(args)

    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    model_without_ddp = model#note model_without_ddp
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    total_batch_size = args.batch_size * args.update_freq * distributed_utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model = torch.nn.parallel.DataParallel(model, device_ids=[args.device])     # 可以改成args.device
    model_without_ddp = model.module #note 为什么要model.module
    print("Start training for %d epochs" % args.epochs) #实现的是 单机/多机-多进程 (Single/Multi-Node Multi-process)
    start_time = time.time()
    min_val_loss = None

    ####base_line_data_enhancement
    ##optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = create_optimizer( #根据args选择优化，args为adam，此为adam
        args, model_without_ddp, skip_list=None,
        get_num_layer= None, filter_bias_and_bn=False,
        get_layer_scale= None)
    print("Use Cosine LR scheduler")  #使用余弦LR调度器
    lr_schedule_values = distributed_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )


    for epoch in range(start_epoch, args.epochs+start_epoch):
        print('epoch',epoch)
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_val.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            log_writer.set_step_epoch(epoch)
        _ ,  Train_pgailv, Train_predict, Train_zqcw, Train_labelindex =train(args, model,data_loader_train,optimizer,device, epoch,
        log_writer=log_writer,start_steps=epoch * num_training_steps_per_epoch,num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        lr_schedule_values = lr_schedule_values,sampler_train=sampler_train
        )

        # 写入pkl文件
        data_pkl_TRAIN = args.data_pkl_addr+'\\TRAIN'
        if not os.path.isdir(data_pkl_TRAIN):
            os.makedirs(data_pkl_TRAIN)

        print('写入pkl文件11111')
        data_pkl_addr1 = data_pkl_TRAIN + '\\'+str(epoch)+'.pkl'
        pkl_content1={'pout':Train_pgailv , 'py':Train_predict , 'zq': Train_zqcw, 'labelindex':Train_labelindex}
        with open(data_pkl_addr1, 'wb') as f:
            print('写入pkl文件')
            pickle.dump(pkl_content1, f)


        ####保存模型
        if args.log_dir and args.save_ckpt:
            if (epoch) % args.save_ckpt_freq == 0 or epoch == args.epochs:
                distributed_utils.save_model(args, epoch, model_without_ddp, optimizer)
        eval_stats, eval_pgailv, eval_predict, eval_zqcw, eval_labelindex = eval(args, model,data_loader_val,optimizer,device, epoch, log_writer=log_writer,start_steps=epoch * num_training_steps_per_epoch,num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq)
        # 写入pkl文件
        data_pkl_TEST = args.data_pkl_addr + '\\TEST'
        if not os.path.isdir(data_pkl_TEST):
            os.makedirs(data_pkl_TEST)
        print('写入pkl文件22222')
        data_pkl_addr2 = data_pkl_TEST + '//' + str(epoch) + '.pkl'
        pkl_content2 = {'pout': eval_pgailv, 'py': eval_predict, 'zq': eval_zqcw,
                       'labelindex': eval_labelindex}
        with open(data_pkl_addr2, 'wb') as f:
            print('写入pkl文件')
            pickle.dump(pkl_content2, f)

        if min_val_loss is None or  min_val_loss >= eval_stats["loss"]:
            min_val_loss = eval_stats["loss"]
            print("Save Best model %d epochs" % epoch)
            if args.log_dir and args.save_ckpt:
                distributed_utils.save_model(args, "best", model_without_ddp, optimizer)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser('Distributed training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.class_names = [str(item) for item in args.class_names.split(',')]
    main(args)