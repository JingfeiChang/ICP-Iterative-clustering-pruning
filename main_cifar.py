import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from utils.options import args
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import utils.common as utils

import os
import copy
import random
import numpy as np
import heapq
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from data import cifar10, cifar100
from importlib import import_module

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
criterion = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg9': 6,
    'vgg16': 13,
    'vgg19': 16,
    'resnet56' : 27,
    'resnet110' : 54,
    'googlenet' : 27,
    'densenet':36,
    }

pruned_food_cfg = {
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 493, 208, 512, 138, 323, 361],
    'vgg19': [45, 37, 110, 24, 113, 16, 51, 23, 96, 108, 72, 238, 47, 211, 151, 44],
    'resnet56': [10, 9, 9, 6, 10, 12, 12, 5, 10, 8, 19, 12, 10, 9, 11, 7, 
                 10, 13, 13, 29, 17, 14, 19, 10, 17, 19, 20],
    'resnet110': [1, 3, 3, 7, 6, 8, 2, 7, 9, 6, 6, 6, 10, 6, 8, 7, 11, 16, 
                  8, 10, 20, 8, 9, 8, 7, 13, 8, 5, 7, 3, 7, 5, 8, 2, 4, 8, 
                  9, 13, 9, 18, 5, 8, 8, 11, 12, 8, 14, 16, 6, 13, 8, 12, 11, 7],
    'googlenet': [96, 16, 32, 128, 32, 96, 96, 16, 48, 112, 24, 64, 128, 24, 64, 144, 32, 64, 
                  160, 32, 128, 160, 32, 128, 192, 48, 128]
    }

original_food_cfg = {
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'vgg19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'resnet56': [16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'resnet110': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                  32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'googlenet': [96, 16, 32, 128, 32, 96, 96, 16, 48, 112, 24, 64, 128, 24, 64, 144, 32, 64, 
                  160, 32, 128, 160, 32, 128, 192, 48, 128]
    }

reg_hook = [3, 5, 6, 10, 12, 13, 17, 19, 20, 24, 26, 27, 31, 33, 34, 
            38, 40, 41, 45, 47, 48, 52, 54, 55, 59, 61, 62] 

food_dimension = conv_num_cfg[args.cfg]
original_food = original_food_cfg[args.cfg]
current_food = pruned_food_cfg[args.cfg]


def load_vgg_particle_model(model, random_rule, oristate_dict):
    #print(ckpt['state_dict'])
    #global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)-1
            orifilter_num1 = oriweight.size(1)
            currentfilter_num1 = curweight.size(1)-1
       

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    # heapq.nlargest(n, iterable[, key]) Return a list with the n largest elements from the dataset defined by iterable
                    # map(function, iterable, ...) 根据提供的函数对指定序列做映射
                    # list.index(x[, start[, end]]) 函数用于从列表中找出某个值第一个匹配项的索引位置。x-- 查找的对象。start-- 可选，查找的起始位置。end-- 可选，查找的结束位置。
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                                # 应该是从原始权重集中继承权重到新的模型
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                select_num = currentfilter_num1
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [0, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    # heapq.nlargest(n, iterable[, key]) Return a list with the n largest elements from the dataset defined by iterable
                    # map(function, iterable, ...) 根据提供的函数对指定序列做映射
                    # list.index(x[, start[, end]]) 函数用于从列表中找出某个值第一个匹配项的索引位置。x-- 查找的对象。start-- 可选，查找的起始位置。end-- 可选，查找的结束位置。
                    select_index.sort()
                for index_i, i in enumerate(select_index):
                    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
                    state_dict[name + '.weight'][:, index_i, :, :] = \
                        oristate_dict[name + '.weight'][:,i, :, :]
                             # 应该是从原始权重集中继承权重到新的模型

                
                last_select_index = None

    model.load_state_dict(state_dict)

def load_google_particle_model(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()
    all_food_conv_name = []
    all_food_bn_name = []

    for name, module in model.named_modules():

        if isinstance(module, Inception):

            food_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            food_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            food_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            food_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4'] #the index of sketch bn weight
            
            for bn_index in food_bn_index:
                all_food_bn_name.append(name + bn_index)

            for weight_index in food_filter_channel_index:
                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    #print(state_dict[conv_name].size())
                    #print(oristate_dict[conv_name].size())
                else:
                    select_index = range(orifilter_num)
         
            
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)


                select_index_1 = copy.deepcopy(select_index)


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                else:
                    select_index = range(orifilter_num)
                
                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                            state_dict[conv_name][index_i][index_j] = \
                                oristate_dict[conv_name][i][j]



            for weight_index in food_channel_index:

                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                #print(state_dict[conv_name].size())
                #print(oristate_dict[conv_name].size())


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()


                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]


            for weight_index in food_filter_index:

                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for index_i, i in enumerate(select_index):
                            state_dict[conv_name][index_i] = \
                                oristate_dict[conv_name][i]


    for name, module in model.named_modules(): #Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_food_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_food_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_resnet_particle_model(model, random_rule, oristate_dict):

    cfg = { 
           'resnet56': [9,9,9],
           'resnet110': [18,18,18],
           }

    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                #logger.info('weight_num {}'.format(conv_weight_name))
                #logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                #logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        #logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]  

                    last_select_index = select_index
                    #logger.info('last_select_index{}'.format(last_select_index)) 

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    #for param_tensor in state_dict:
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


# Data
print('==> Loading Data..')
if args.dataset == 'cifar10':
    loader = cifar10(args)

elif args.dataset == 'cifar100':
    loader = cifar100(args)


# Model
print('==> Loading Model..')
if args.arch == 'vgg':
    #model_t = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    model_t = import_module(f'model.{args.mmd}').vggmmd(args.cfg, foodsource=original_food).to(device)   # object detection
elif args.arch == 'resnet':
    model_t = import_module(f'model.{args.arch}').resnet(args.cfg, food=original_food).to(device)
elif args.arch == 'googlenet':
    model_t = import_module(f'model.{args.arch}').googlenet().to(device)

ckpt_t = torch.load(args.teacher_dir, map_location=device)
state_dict_t = ckpt_t['state_dict']
model_t.load_state_dict(state_dict_t)



fmap_block = []
def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)

# Training
def train(model_s, model_t, optimizers, trainLoader, args, epoch):
    
    model_s.train()
    losses_kd = utils.AverageMeter()       # knowledge distillation loss
    losses_at = utils.AverageMeter()       # attention transfer loss
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    

    
    optimizer_s = optimizers[0]

    num_iterations = len(trainLoader)    #391
    


    for i, (inputs, targets) in enumerate(trainLoader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        ############################
        # (1) Update student network
        ###########################
           
        features_t, attention_t = model_t(inputs)       
        features_s, attention_s = model_s(inputs)

        optimizer_s.zero_grad()
        
        '''
        # MSE Loss of two output features
        error_data = args.miu * criterion(features_s, targets)

        losses_data.update(error_data.item(), inputs.size(0))   # dict.update(dict2)  把字典dict2的键/值对更新到dict里
        error_data.backward(retain_graph=True)
        '''

        # Knowledge Distillation Loss
        
        error_kd = utils.distillation(features_s, features_t, targets, args.temperature, args.alpha)
        losses_kd.update(error_kd.item(), inputs.size(0))
        error_kd.backward(retain_graph=True)        
        
        
        # attention Loss      
        at_loss_groups = [utils.at_loss(x, y) for x, y in zip(attention_s, attention_t)]
        at_loss_groups = [v.sum() for v in at_loss_groups]

        error_at = 1 * sum(at_loss_groups)
        losses_at.update(error_at.item(), inputs.size(0))
        error_at.backward(retain_graph=True)


        #*********************************************************************************#

        optimizer_s.step()


        prec1, prec5 = utils.accuracy(features_s, targets, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if i % args.print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}):\t'
                'Loss_kd {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'
                'Loss_at {loss_at.val:.4f} ({loss_at.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, num_iterations, loss_kd=losses_kd, loss_at=losses_at, top1=top1, top5=top5))        
# Testing
def test(model_s, testLoader):
    
    model_s.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()
    

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testLoader):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_s(inputs)[0]
            loss = cross_entropy(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
        
        logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))
       
    return top1.avg, top5.avg
# Pruning

def Pruning(model, trainLoader):
    print('==> Start Pruning..')   
    netchannels=[]
    retain_c = []    
    a = 0
    # register hook        
    if args.arch == 'vgg':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                
                a = a+1
                if a <= 13:
                
                    handle = m.register_forward_hook(forward_hook)
                
                #handle = m.register_forward_hook(forward_hook)
    elif args.arch == 'resnet':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                a = a+1
                if a % 2 == 0:
                    handle = m.register_forward_hook(forward_hook)   
    elif args.arch == 'googlenet':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                a = a+1
                if a in reg_hook:
                    handle = m.register_forward_hook(forward_hook)    

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(trainLoader):
            if i == 1:
                inputs = inputs.to(device)
                model(inputs)

    channels = conv_num_cfg[args.cfg]      
    netchannels = torch.zeros(channels)
    for s in range(channels):

        # change the size of fmap_block from (batchsize, channels, W, H) to (batchsize, channels, W*H)
        a, b, c, d = fmap_block[s].size()
        fmap_block[s] = fmap_block[s].view(a, b, -1)

        fmap_block[s] = torch.sum(fmap_block[s], dim=0)/a

        
        # clustering
        X = np.array(fmap_block[s].cpu())
        clustering = DBSCAN(eps=0.002, min_samples=5, metric='cosine').fit(X)
        
        # defult: eps=0.5, min_samples=5
        # ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
        labels = clustering.labels_

        #print(labels)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
    
        netchannels[s] = netchannels[s]+n_clusters_+n_noise_

        #print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)
        retain_c.append(int(netchannels[s]))       

    handle.remove()  
    fmap_block.clear()
    print(retain_c) 
    return retain_c

gbest_state = {}

path = args.prune_dir

def main():
    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    if args.arch == 'vgg':
        #model_s = import_module(f'model.{args.arch}').ATGAPVGG(args.cfg, foodsource=current_food).to(device)
        model_s = import_module(f'model.{args.mmd}').vggmmd(args.cfg, foodsource=current_food).to(device)
    elif args.arch == 'resnet':
        model_s = import_module(f'model.{args.arch}').resnet(args.cfg, food=current_food).to(device)
    elif args.arch == 'googlenet':
        model_s = import_module(f'model.{args.arch}').googlenet(food=original_food).to(device) 
       
    print(model_s)    
    #model_dict_s = model_s.state_dict()
    #model_dict_s.update(state_dict_t)
    #model_s.load_state_dict(model_dict_s)
   
    test(model_t, loader.loader_test)


    param_s = [param for name, param in model_s.named_parameters()]
    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #scheduler_s = MultiStepLR(optimizer_s, milestones=args.lr_decay_step, gamma=0.1)
    scheduler_s = CosineAnnealingLR(optimizer_s, T_max=202)
    
    optimizers = [optimizer_s]
    schedulers = [scheduler_s]

    for epoch in range(start_epoch, args.num_epochs):
        
        print('=> training')
        
        train(model_s, model_t, optimizers, loader.loader_train, args, epoch)
        
        for s in schedulers:
            s.step()
            lr = s.get_lr()
            print(lr)       
        
        test_prec1, test_prec5 = test(model_s, loader.loader_test)
        
        print('**************************************************************************')
        
        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        model_state_dict = model_s.module.state_dict() if len(args.gpus) > 1 else model_s.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer_s': optimizer_s.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, is_best)    
        '''
        if epoch <= 20 and epoch % 10 == 0:           
            checkpt = torch.load(args.prune_dir, map_location=device)
            retain = Pruning(model_s, loader.loader_train)        
            if args.arch == 'vgg':
                #model_s = import_module(f'model.{args.arch}').ATGAPVGG(args.cfg, foodsource=retain).to(device) 
                model_s = import_module(f'model.{args.mmd}').vggmmd(args.cfg, foodsource=retain).to(device)
                oristate_dict = checkpt['state_dict']
                load_vgg_particle_model(model_s, args.random_rule, oristate_dict)
            elif args.arch == 'resnet':
                model_s = import_module(f'model.{args.arch}').resnet(args.cfg,food=retain).to(device)
                oristate_dict = checkpt['state_dict']
                load_resnet_particle_model(model_s, args.random_rule, oristate_dict) 
            elif args.arch == 'googlenet':
                model_s = import_module(f'model.{args.arch}').googlenet(food=retain).to(device)  

            epoch = checkpt['epoch']
            
            param_s = [param for name, param in model_s.named_parameters()]
            optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            #scheduler_s = MultiStepLR(optimizer_s, milestones=args.lr_decay_step, gamma=0.1)
            scheduler_s = CosineAnnealingLR(optimizer_s, T_max=10)
            
            optimizers = [optimizer_s]
            schedulers = [scheduler_s]      
        '''
    #print(model_s)    
    logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")



if __name__ == '__main__':
    main()
