# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:31:46 2021

@author: cjf_h
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from utils.options_imagenet import args
from sklearn import preprocessing
import utils.common as utils

import os
import copy
import random
import numpy as np
import heapq
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from model import Discriminator
from data import imagenet
from importlib import import_module

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
criterion = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg16': 13,
    'resnet18': 8,
    'resnet34': 16,
    'resnet50': 16,
}

original_food_cfg = {
    'resnet18': [64, 64, 128, 128, 256, 256, 512, 512],
    'resnet34': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet50': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
}

food_dimension = conv_num_cfg[args.student_model]
original_teacher_food = original_food_cfg[args.teacher_model]
original_student_food = original_food_cfg[args.student_model]
 
#load pre-train params
def load_vgg_particle_model(model, random_rule, oristate_dict):
    # print(ckpt['state_dict'])
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (
                    random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num - 1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_resnet_particle_model(model, random_rule, oristate_dict):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    state_dict = model.state_dict()

    current_cfg = cfg[args.student_model]
    last_select_index = None

    all_food_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            if args.student_model == 'resnet18' or args.student_model == 'resnet34':
                iter = 2  # the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            for l in range(iter):
                conv_name = 'module.' + layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_food_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                # logger.info('weight_num {}'.format(conv_weight_name))
                # logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                # logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index
                    # logger.info('last_select_index{}'.format(last_select_index))

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
            if conv_name not in all_food_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    # for param_tensor in state_dict:
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    # for param_tensor in model.state_dict():
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))

    model.load_state_dict(state_dict)

# Data
print('==> Loading Data..')
loader = imagenet(args)

# Model
print('==> Loading Model..')

if args.arch == 'vgg_imagenet':
    model_t = import_module(f'model.{args.arch}').VGG(args.teacher_model).to(device)
elif args.arch == 'resnet_imagenet':
    model_t = import_module(f'model.{args.arch}').resnet(args.teacher_model, food=original_teacher_food).to(device)

ckpt_t = torch.load(args.teacher_dir, map_location=device)
state_dict_t = ckpt_t['state_dict']
model_t.load_state_dict(state_dict_t)



model_d = Discriminator().to(device) 

pruned_state = {}

fmap_block = []
def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)

# Training
def train(model_s, model_d, model_t, optimizers, trainLoader, args, epoch):
    
    model_s.train()
    model_d.train()
    losses_d = utils.AverageMeter()        # discriminator loss
    losses_kd = utils.AverageMeter()       # knowledge distillation loss
    losses_g = utils.AverageMeter()        # GAN loss
    losses_at = utils.AverageMeter()       # attention transfer loss
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    bce_logits = nn.BCEWithLogitsLoss()    #Sigmoid +BCELoss
    
    optimizer_d = optimizers[0]
    optimizer_s = optimizers[1]

    num_iterations = len(trainLoader)    #391
    
    real_label = 1
    fake_label = 0

    for i, (inputs, targets) in enumerate(trainLoader):

        inputs = inputs.to(device)
        targets = targets.to(device)

        ############################
        # (1) Update D network
        ###########################
        
        for p in model_d.parameters():  
            p.requires_grad = True  

        optimizer_d.zero_grad()
        
        features_t, attention_t = model_t(inputs)       
        features_s, attention_s = model_s(inputs)

        output_t = model_d(features_t.detach())   #通过.detach() “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
        # print('output_t',output_t)
        labels_real = torch.full_like(output_t, real_label, device=device)   #torch.full_like(input, fille_value)，就是将input的形状作为返回结果tensor的形状。
        error_real = bce_logits(output_t, labels_real)

        output_s = model_d(features_s.detach())
        # print('output_s',output_t)
        labels_fake = torch.full_like(output_s, fake_label, device=device)
        error_fake = bce_logits(output_s, labels_fake)

        error_d = error_real + error_fake

        #labels = torch.full_like(output_s, real_label, device=device)  # yuanlibuming
        #error_d += bce_logits(output_s, labels)

        error_d.backward()
        losses_d.update(error_d.item(), inputs.size(0))
        
        optimizer_d.step()
        
        ############################
        # (2) Update student network
        ###########################

        for p in model_d.parameters():  
            p.requires_grad = False  

        optimizer_s.zero_grad()

        # Knowledge Distillation Loss
        
        error_kd = utils.distillation(features_s, features_t, targets, args.temperature, args.alpha)
        losses_kd.update(error_kd.item(), inputs.size(0))
        error_kd.backward(retain_graph=True)        
        
        
        # attention Loss      
        at_loss_groups = [utils.at_loss(x, y) for x, y in zip(attention_s, attention_t)]
        at_loss_groups = [v.sum() for v in at_loss_groups]

        error_at = 50 * sum(at_loss_groups)
        losses_at.update(error_at.item(), inputs.size(0))
        error_at.backward(retain_graph=True)
        

        # fool discriminator
        output_s = model_d(features_s.to(device))
        
        labels = torch.full_like(output_s, real_label, device=device)
        error_g = bce_logits(output_s, labels)

        losses_g.update(error_g.item(), inputs.size(0))
        error_g.backward(retain_graph=True)

        #*********************************************************************************#

        optimizer_s.step()


        prec1, prec5 = utils.accuracy(features_s, targets, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if i % args.print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}):\t'
                'Loss_kd {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'
                'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                'Loss_g {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                'Loss_at {loss_at.val:.4f} ({loss_at.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, num_iterations, loss_kd=losses_kd, loss_d=losses_d, 
                loss_g=losses_g, loss_at=losses_at, top1=top1, top5=top5))

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
    retain_c = []    
    a = 0
    # register hook        
    if args.arch == 'vgg':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                handle = m.register_forward_hook(forward_hook)
    elif args.arch == 'resnet_imagenet':          
        reg_hook = [2, 4, 6, 9, 11, 14, 16, 19]   # resnet18
        #reg_hook = [2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34]  # resnet34
        i = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                i = i + 1
                if i in reg_hook:
                    handle = m.register_forward_hook(forward_hook)     

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(trainLoader):
            if i == 1:
                inputs = inputs.to(device)
                model(inputs)

    channels = conv_num_cfg[args.student_model]
     
    for s in range(channels):

        # change the size of fmap_block from (batchsize, channels, W, H) to (batchsize, channels, W*H)
        re_channel = 0
        a, b, c, d = fmap_block[s].size()
        fmap_block[s] = fmap_block[s].view(a, b, -1)

        fmap_block[s] = torch.sum(fmap_block[s], dim=0)/a

        
        # change the size of fmap_block from (channels, W*H) to (channels, 1)
        fmap_block[s] = torch.sum(torch.abs(fmap_block[s]), dim=1).cpu()
                
        #print(fmap_block[s])

        fmap_block[s] = preprocessing.Normalizer(norm='max').fit_transform(fmap_block[s].reshape(1, -1))
        fmap_block[s] = np.squeeze(fmap_block[s])
        fmap_block[s] = torch.Tensor(fmap_block[s])
                
        #print(fmap_block[s])
                
        imscore_copy = fmap_block[s]
        mask = imscore_copy.gt(args.thre).float().cuda()
                
        if int(torch.sum(mask)) <= 1:
            re_channel = 1
        else:
            re_channel = int(torch.sum(mask))
        retain_c.append(re_channel)      
            
    handle.remove()  
    fmap_block.clear()
    print(retain_c) 
    return retain_c

def main():
    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    if args.arch == 'vgg':
        model_s = import_module(f'model.{args.arch}').ATGAPVGG(args.student_model, foodsource=original_student_food).to(device)
    elif args.arch == 'resnet':
        model_s = import_module(f'model.{args.arch}').resnet(args.student_model, food=original_student_food).to(device)



    ckpt_s = torch.load(args.student_dir, map_location=device)
    state_dict_s = ckpt_s['state_dict']
    model_s.load_state_dict(state_dict_s)

    
    if len(args.gpus) != 1:
        model_s = nn.DataParallel(model_s, device_ids=args.gpus)

    test(model_s, loader.loader_test)


    optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    param_s = [param for name, param in model_s.named_parameters()]
    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler_d = MultiStepLR(optimizer_d, milestones=args.lr_decay_step, gamma=0.1)
    #scheduler_d = CosineAnnealingLR(optimizer_d, T_max=80)
    scheduler_s = MultiStepLR(optimizer_s, milestones=args.lr_decay_step, gamma=0.1)
    #scheduler_s = CosineAnnealingLR(optimizer_s, T_max=80)
    
    optimizers = [optimizer_d, optimizer_s]
    schedulers = [scheduler_d, scheduler_s]

    for epoch in range(start_epoch, args.num_epochs):
        
        print('=> training')
        
        train(model_s, model_d, model_t, optimizers, loader.loader_train, args, epoch)
        
        for s in schedulers:
            s.step()
            lr = s.get_last_lr()
            print(lr)       
        
        test_prec1, test_prec5 = test(model_s, loader.loader_test)
        
        print('**************************************************************************')
        
        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        model_state_dict = model_s.module.state_dict() if len(args.gpus) > 1 else model_s.state_dict()

        state = {
            'state_dict_s': model_state_dict,
            'state_dict_d': model_d.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_s': optimizer_s.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_s': scheduler_s.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, is_best)    
        
        if epoch <= 16 and epoch % 4 == 0:           
            checkpt = torch.load(args.prune_dir, map_location=device)
            retain = Pruning(model_s, loader.loader_train)        
            if args.arch == 'vgg':
                model_s = import_module(f'model.{args.arch}').ATGAPVGG(args.student_model, foodsource=retain).to(device) 
                oristate_dict = checkpt['state_dict_s']
                load_vgg_particle_model(model_s, args.random_rule, oristate_dict)
            elif args.arch == 'resnet':
                model_s = import_module(f'model.{args.arch}').resnet(args.student_model,food=retain).to(device) 
                load_resnet_particle_model(model_s, args.random_rule, oristate_dict)

            epoch = checkpt['epoch']
            
            optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            param_s = [param for name, param in model_s.named_parameters()]
            optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            scheduler_d = MultiStepLR(optimizer_d, milestones=args.lr_decay_step, gamma=0.1)
            #scheduler_d = CosineAnnealingLR(optimizer_d, T_max=80)
            scheduler_s = MultiStepLR(optimizer_s, milestones=args.lr_decay_step, gamma=0.1)
            #scheduler_s = CosineAnnealingLR(optimizer_s, T_max=80)
            
            optimizers = [optimizer_d, optimizer_s]
            schedulers = [scheduler_d, scheduler_s]          
    print(model_s)    
    logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

if __name__ == '__main__':
    main()
