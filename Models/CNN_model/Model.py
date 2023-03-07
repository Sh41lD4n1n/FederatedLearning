'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import os
import argparse

from models import *
from utils import progress_bar


from torch.optim import Optimizer

os.chdir("..")
from StatisticClass import Statistic
os.chdir("CNN_model")

class Optimizer_SGD(Optimizer):
    
    def __init__(self, params, lr=1e-2):
        super(Optimizer_SGD, self).__init__(params, defaults={'lr': lr})
        #self.momentum = momentum
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict()#dict(mom=torch.zeros_like(p.data))
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self.state:
                    self.state[p] = dict()#dict(mom=torch.zeros_like(p.data))
                #mom = self.state[p]['mom']
                #mom = self.momentum * mom - group['lr'] * p.grad.data
                
                #self.w = self.w - M_scaller@(self.lr*(precond_value@der))
                p.data -= group['lr']*p.grad.data


"""
init model

model = Model("name")
model.set_optimizer(opt)
model.set_preconditioner(preconditioner)
model.set_scaller(scaller)

"""


class Model:
    def __init__(self,name):
        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.select_model()

        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = Optimizer_SGD()
        
        
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.best_acc = 0
        self.current_acc = 0
        
        self.current_epoch = 0
        self.stat_collector = Statistic(name)
        

    def set_optimizer(self,opt):
        self.optimizer = opt
    
    def set_preconditioner(self,preconditioner):
        self.optimizer.preconditioner = preconditioner
    
    def set_scaller(self,scaller):
        self.optimizer.scaller = scaller
    
    def select_model(self):
        # net = VGG('VGG19')
        net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        # net = SimpleDLA()
        net = net.to(self.device)
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        
        return net
    
    def recover_from_checkpoints(self):
        # Load checkpoint.
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        assert os.path.exists('./checkpoint/{self.name}_ckpt.pth'), 'Error: no checkpoints for model {self.name}!'
        checkpoint = torch.load(f'./checkpoint/{self.name}_ckpt.pth')
        
        self.net.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
    
    def save_checkpoints(self):
        state = {
            'net': self.net.state_dict(),
            'acc': self.current_acc,
            'epoch': self.current_epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{self.name}_ckpt.pth')
    
    def init_model():
        pass
    
    def train(self,trainloader):
        #print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(self,testloader):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        self.stat_collector.handle_test(loss=test_loss,accuracy=correct/total)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            self.best_acc = acc 
            self.save_checkpoints()
            
    def set_parameters(self,params_dict):
        self.net.load_state_dict(params_dict)

    def get_parameters(self):
        return self.net.state_dict()

    def run(self,epoch,train_loader,test_loader):
        for current_epoch in range(self.current_epoch, self.current_epoch+epoch):
            self.train(train_loader)
            self.test(test_loader)
            #self.scheduler.step()

