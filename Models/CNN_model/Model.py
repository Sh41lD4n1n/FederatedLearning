'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.nn.init as init

import numpy as np

import os
import argparse

from models import *
from utils import progress_bar

from tqdm import tqdm


import torch.optim as optim
from torch.optim import Optimizer

os.chdir("..")
from StatisticClass import Statistic
os.chdir("CNN_model")

from torch.nn.utils import _stateless
from functorch import hessian



class Model:
    """
    Model:
    - train/test модель
        - запись и получение весов модели
        - сохранение параметров (пока не применяется)
    """
    
    def __init__(self,name):
        """
        Инициализация
        Параметры:
            name   -    имя эксперимента, учитывающее номер worker-а nodы
                   (пример: "IdenticalSplit_singleWorker_#number")
            
            device -    Cpu/Gpu
            net    -    Neural Network
            criterion-  функция потерь
            optimizer-  Оптимизатор
            scheduler-  optimizer learning rate scheduler (not used)

            best_acc -  Наибольшее значение точности
            current_acc Текущее значение точности
            
            current_epoch   текущая эпоха/шаг
            stat_collector  колектор точности/потерь для train/test
            
        - настройка параметров
        """
        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.select_model()

        self.criterion = nn.CrossEntropyLoss()
        
        self.is_oasis = False

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)#Optimizer_SGD(params = self.net.parameters())
        
        
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        
        self.current_epoch = 0
        self.stat_collector = Statistic(name)
        
        

    def set_optimizer(self,opt,is_oasis=False):
        """
        Запись optimizer
        """
        self.optimizer = opt
        self.is_oasis = is_oasis
    
    def oasis_preprocess(self,second_derivative):
        if self.is_oasis:
            print(second_derivative,second_derivative.shape)
            self.optimizer.set_der(second_derivative)
    
    def select_model(self):
        """
        select_model: Выбор и настройка модели
        - всегда брать ResNet18
        - загрузка модели на gpu
        """

        #net = VGG('VGG19')
        #net = ResNet18()
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
        net = LeNet()
        net = net.to(self.device)
        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        
        return net
    

    def train(self,trainloader):
        """
        Тренировка модели (взято из данного github репозитория)
        
        - net модели после итерации возращает "вероятности"
        принадлежности к каждому классу
        - max(1) возращает класс с наибольшей вероятностью
        для каждого элемента
        """

        #print('\nEpoch: %d' % epoch)
        self.net.to(self.device)
        self.net.train()
        #loss для одной epoch
        train_loss = 0
        #подсчет точность
        correct = 0
        total = 0
        #применить для каждого бача из trainload
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #тренировка
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            #self.stat_collector.writer.add_graph(self.net,inputs)
            outputs = self.net(inputs)
            #шаг оптимизации
            loss = self.criterion(outputs, targets)
            
            
            # if self.is_oasis:
            #     loss.backward(create_graph=True)
            # else:
            #     loss.backward()
            loss.backward()


            def model_run(params):
                names = list(n for n, _ in self.net.named_parameters())
                #y_hat = _stateless.functional_call(self.net, {n: p for n, p in zip(names, params)}, inputs)
                return self.criterion(outputs, targets)

            if self.is_oasis:
                second_derivative = hessian(model_run)(tuple(self.net.parameters()) ) # torch.autograd.functional.hessian
                self.oasis_preprocess(second_derivative = second_derivative)

            
            self.optimizer.step()

            train_loss += loss.item()

            #подсчет точности
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # функция вывода состояния модели точность, loss
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # запись loss/acc для train в классе statistic, и в Tensorboard
        self.stat_collector.handle_train(loss=train_loss/(batch_idx+1),accuracy=correct/total,weights = list(self.net.cpu().parameters())[0])


    def test(self,testloader):
        """
        Тест модели (взято из данного github репозитория)
        """
        self.net.to(self.device)
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

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # запись loss/acc для test в классе statistic, и в Tensorboard
        self.stat_collector.handle_test(loss=test_loss/(batch_idx+1),accuracy=correct/total)

        # Save checkpoint.
        #acc = 100.*correct/total
        #if acc > self.best_acc:
        #    self.best_acc = acc 
        #    self.save_checkpoints()
            
    
    
    def set_parameters(self,params):
        """
        set_parameters
            Параметры:
                - params - новые параметры для записи
            
            Записывает параматры из `params` в модель,
            итерируясь по параметрам  модели и новым значениям
        """

        with torch.no_grad():
            for p_new,p in zip(params,self.net.parameters()):
                p_new = p_new.to(self.device)
                p = p.to(self.device)
                new = torch.nn.parameter.Parameter(p_new.data.clone())
                new = new.to(self.device)
                p.data = new #p.data + new - p.data


    
    def get_parameters(self):
        """ get_parameters
            Возвращает параматры модели,превращая 
            итериратор в лист
        """
        return list(self.net.cpu().parameters())


    def run(self,epoch,train_loader,test_loader):
    	for current_epoch in range(0, epoch):
    	    self.train(train_loader)
    	    self.test(test_loader)
	    #self.scheduler.step()

