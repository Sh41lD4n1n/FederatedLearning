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


from torch.optim import Optimizer

os.chdir("..")
from StatisticClass import Statistic
os.chdir("CNN_model")

"""
Optimizer_SGD - собственная имплементация 
SGD оптимизатора
(пробовал использовать и Optimizer_SGD и встроенную версию SGD)
"""
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
Model:
- train/test модель
    - запись и получение весов модели
    - сохранение параметров (пока не применяется)
"""
class Model:
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
    def __init__(self,name):
        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.select_model()

        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = Optimizer_SGD(params = self.net.parameters())
        
        
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.best_acc = 0
        self.current_acc = 0
        
        self.current_epoch = 0
        self.stat_collector = Statistic(name)
        
    """
    Запись optimizer
    """
    def set_optimizer(self,opt):
        self.optimizer = opt
    
    """
    Запись preconditioner (пока не применяется)
    """
    def set_preconditioner(self,preconditioner):
        self.optimizer.preconditioner = preconditioner
    
    """
    Запись scaller (scheduler) (пока не применяется)
    """
    def set_scaller(self,scaller):
        self.optimizer.scaller = scaller
    
    """
    select_model: Выбор и настройка модели
    - всегда брать ResNet18
    - загрузка модели на gpu
    """
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
    
    """
    Запись параметров и востановление: (сейчас не используется)
    recover_from_checkpoints
    save_checkpoints
    """
    def recover_from_checkpoints(self):
        # Load checkpoint.
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        assert os.path.exists(f'./checkpoint/{self.name}_ckpt.pth'), f'Error: no checkpoints for model {self.name}!'
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
        torch.save(state, f'./checkpoint/{self.name}_ckpt.pth')
    
    """
    Инициализация параметров модели (сейчас не используется)
    с функцией weight_init
    """
    def init_model(self):
        return
        #self.net.apply(weight_init)
    
    """
    Тренировка модели (взято из данного github репозитория)
    
    - net модели после итерации возращает "вероятности"
    принадлежности к каждому классу
    - max(1) возращает класс с наибольшей вероятностью
    для каждого элемента
    """
    def train(self,trainloader):
        #print('\nEpoch: %d' % epoch)
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
            outputs = self.net(inputs)
            #шаг оптимизации
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            #подсчет точности
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # функция вывода состояния модели точность, loss
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # запись loss/acc для train в классе statistic, и в Tensorboard
        self.stat_collector.handle_train(loss=train_loss/(batch_idx+1),accuracy=correct/total,weights = list(self.net.parameters())[0])

    """
    Тест модели (взято из данного github репозитория)
    """
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
        
        # запись loss/acc для test в классе statistic, и в Tensorboard
        self.stat_collector.handle_test(loss=test_loss/(batch_idx+1),accuracy=correct/total)

        # Save checkpoint.
        #acc = 100.*correct/total
        #if acc > self.best_acc:
        #    self.best_acc = acc 
        #    self.save_checkpoints()
            
    """
    Пробовал обмен параметрами через
        .parameters() (такой метод использует оптимизаторы)
        .load_state_dict() (применяется для сохраниения весов в файл)
    """
    """
    set_parameters
        Параметры:
            - params - новые параметры для записи
        
        Записывает параматры из `params` в модель,
        итерируясь по параметрам  модели и новым значениям
    """
    #"""
    def set_parameters(self,params):
        with torch.no_grad():
            for p_new,p in zip(params,self.net.parameters()):
                p.data = p_new.data.clone()
    #"""
    
    """
    get_parameters
        
        Возвращает параматры модели,превращая 
        итериратор в лист
    """
    #"""
    def get_parameters(self):
        return list(self.net.parameters())
    #"""

    """
    set_parameters
        (один из способов обработки параметров,
         был проверен, не сработал,
         буду пререпроверять)

        Параметры:
            - params_dict - новые параметры для записи
        
        Записывает параматры из словаря `params_dict` в модель,
        через load_state_dict
        (принимает словарь key - имя слой, value - параметры)
    """
    """
    def set_parameters(self,params_dict):
        self.net.load_state_dict(params_dict)
    """
    """
    get_parameters
        (один из способов обработки параметров,
         был проверен, не сработал,
         буду пререпроверять)
        
        Возращает параматры из словаря `params_dict` в модель,
        через load_state_dict
        (принимает словарь key - имя слой, value - параметры)
    """
    """
    def get_parameters(self):
        return self.net.state_dict()
    """

    def run(self,epoch,train_loader,test_loader):
        for current_epoch in range(self.current_epoch, self.current_epoch+epoch):
            self.train(train_loader)
            self.test(test_loader)
            #self.scheduler.step()

"""
weight_init (сейчас не применяется)
    Параметры:
        m: NN модель

    - Инициализирует параметры модели
    Usage:
            model = Model()
            model.apply(weight_init)
"""
def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    else:
        raise Exception("Non initialization")
