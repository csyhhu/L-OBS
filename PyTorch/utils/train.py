"""
This file contains some utility functions for using ImageNet dataset and L-OBS

Author: Chen Shangyu (schen025@e.ntu.edu.sg)

"""
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim

import collections
from datetime import datetime

import numpy as np

import tensorflow as tf

import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pickle


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(model, val_loader, val_record, train_record, n_batch_used=100, use_cuda=True):

    if n_batch_used == -1:
        n_batch_used = len(val_loader)

    monitor_freq = int(n_batch_used / 5)

    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    # end = time.time()
    for i, (inputs, target) in enumerate(val_loader):

        if use_cuda:
            target = target.cuda()
            inputs = inputs.cuda()

        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if i % monitor_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), loss=losses,
                top1=top1, top5=top5))

        if i == n_batch_used:
            break

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if val_record is not None:
        val_record.write('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                         .format(top1=top1, top5=top5))
    if train_record is not  None:
        train_record.write('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                           .format(top1=top1, top5=top5))

    model.train()

    return top1.avg, top5.avg


def adjust_mean_var(net, train_loader, train_file, n_batch_used=500, use_cuda=True):
    monitor_freq = int(n_batch_used / 5)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    net.train()

    # end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = net(input_var)

        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        if (i) % monitor_freq == 0:
            print('Train: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, n_batch_used, loss=losses,
                top1=top1, top5=top5))
            if train_file != None:
                train_file.write('[%d/%d] Loss: %f, Prec@1: %f, Prec@5: %f\n' % \
                                 (i, n_batch_used, losses.avg, top1.avg, top5.avg))

        if (i) == n_batch_used:
            break