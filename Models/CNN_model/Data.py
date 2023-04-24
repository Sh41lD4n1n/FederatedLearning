"""
data = Data(split="ident",n_workers=5)
data.get_data_loaders()
"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset,Dataset

import random
"""
Data класс
    хранит CIFAR dataset,
    делит его на части для nodes(workers)
    и хранит dataloader и torch dataset.

"""  

class MySubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = []
        self._init_targets()
        #self.transform = transform

    def _init_targets(self):
        self.targets = []
        for i in range(self.__len__()):
            _,label = self.__getitem__(i)
            self.targets.append(label)
        pass

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        return im,labels #self.transform(im), labels

    def __len__(self):
        return len(self.indices)

class Data:
    SPLITS = ["het","ident","random_indent", "ident_same","noSplit","noSplit_test_split"]
    """
    Инициализация объекта класса Data:
        Параметры:
            split: тип хранения данных из SPLITS:
                "het" - количество частей n_workers, колличество изображений
                        из каждого класса распределено не равномерно по частям
                "ident" - количество частей n_workers, колличество изображений
                          из каждого класса распределено равномерно по частям
                "noSplit" - одна часть которая хранит все изображения
        - Хранит dataset
        - вызывает функцию загрузки dataset
        - вызывает функцию деления dataset
    """
    def __init__(self,split,n_workers,environment):
        # массив из Dataset объектов для тренировки и проверок
        self.trainset = []
        self.testset = []

        #сет из имен классов и индексов классов
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
        self.classes_number = [i for i in range(0,10)]

        # ининциализация: загрузить dataset из torchvision
        # и применить преобразования изображения
        

        self._initialize(environment)

        # 
        self._split(n_workers=n_workers,split_type=split)
        
            
    """
    Загрузка CIFAR
    - Загружает train, test
    - Применяет обработку данных обрезка, поворот, нормализация
    """
    def _initialize(self,environment):
        # Создание функций обработки для train/test
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Загрузка данных директория зависит от места запуска
        if environment == "kaggle":
            self.trainset = [torchvision.datasets.CIFAR10(
                root='/kaggle/input/federatedlearning/Models/CNN_model/data/cifar-10-python', train=True, download=False, transform= transform_train)]
            
            self.testset = [torchvision.datasets.CIFAR10(
                root='/kaggle/input/federatedlearning/Models/CNN_model/data/cifar-10-python', train=False, download=False, transform=transform_test)]
        else:
            self.trainset = [torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform= transform_train)]

            self.testset = [torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform= transform_test)]
        

    def get_data_loaders(self):

        dataloader = []
        for trainset,testset in zip(self.trainset,self.testset):
            trainloader = (
                torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=2))
            
            testloader = (
                torch.utils.data.DataLoader(
                testset, batch_size=128, shuffle=True, num_workers=2))
            
            dataloader.append({"trainloader":trainloader,"testloader":testloader})
        
        return dataloader
    
    def get_data(self):
        return self.trainset,self.testset

    def _split(self,n_workers,split_type):
        assert len(self.trainset) == 1 and len(self.testset) == 1, 'Error: Data have been splited before'

        targets_array_train = np.array(self.trainset[0].targets)
        targets_array_test = np.array(self.testset[0].targets)
        
        
        workers_idexes_train = []
        workers_idexes_test = []
        
        if split_type =="het":
            workers_idexes_train = self._get_heterogenious_split(n_workers=n_workers,targets_array=targets_array_train)
            workers_idexes_test = [np.arange(len(self.testset[0].targets)) for i in range(n_workers)] #self._get_identical_v2_split_index(n_workers=n_workers,targets_array=targets_array_test)
        elif split_type =="ident":
            workers_idexes_train = self._get_identical_split_index(n_workers=n_workers,targets_array=targets_array_train)
            workers_idexes_test = [np.arange(len(self.testset[0].targets)) for i in range(n_workers)] #self._get_identical_v2_split_index(n_workers=n_workers,targets_array=targets_array_test)
        elif split_type =="ident_same":
            workers_idexes_train = self._get_identical_v2_split_index(n_workers=n_workers,targets_array=targets_array_train)
            workers_idexes_test = [np.arange(len(self.testset[0].targets)) for i in range(n_workers)]
        elif split_type =="random_indent":
            workers_idexes_train = self._get_random_split_index(n_workers=n_workers,targets_array=targets_array_train)
            workers_idexes_test = [np.arange(len(self.testset[0].targets)) for i in range(n_workers)]
        elif split_type =="noSplit_test_split":
            workers_idexes_train = [[i for i in range(len(targets_array_train))]]
            workers_idexes_test = self._get_identical_v2_split_index(n_workers=n_workers,targets_array=targets_array_test)
        else:
            return
            workers_idexes_train = [targets_array_train]
            workers_idexes_test = [targets_array_test]


        new_dataset_train = []
        new_dataset_test = []
        for w_train,w_test in zip(workers_idexes_train,workers_idexes_test):
            new_dataset_train.append(MySubset(self.trainset[0],w_train))
            new_dataset_test.append(MySubset(self.testset[0],w_test))
        
        self.trainset,self.testset = new_dataset_train.copy(),new_dataset_test.copy()

    def _get_random_split_index(self,n_workers,targets_array):
        n_images = len(targets_array)
        images_array = np.random.choice(n_images,n_images, replace=False)
        workers_idexes = [[] for _ in range(n_workers)]
        n_images_per_worker = n_images//n_workers

        for w in range(n_workers):
            workers_idexes[w] = list(images_array[w*n_images_per_worker:(w+1)*n_images_per_worker])
        return workers_idexes

    def _get_identical_v2_split_index(self,n_workers,targets_array):
        """
        Each workers has same data from worker 1
        """
        workers_idexes = self._get_identical_split_index(n_workers,targets_array)
        
        worker_idexes = workers_idexes[0].copy()
        workers_idexes = [worker_idexes for _ in range(n_workers)]

        return workers_idexes

    def _get_identical_split_index(self,n_workers,targets_array):

        (values,counts) = np.unique(targets_array, return_counts=True)
        workers_idexes = [[] for _ in range(n_workers)]

        for i,count in zip(values,counts):

            indxes = np.arange(len(targets_array))[targets_array == i]
            amount = count//n_workers

            init_pos = 0
            for w in range(n_workers):
                workers_idexes[w] += list(indxes[init_pos:init_pos+amount])
                init_pos += amount

        for w in range(n_workers):
            random.shuffle(workers_idexes[w])
        return workers_idexes
    
    def _get_heterogenious_split(self, targets_array, n_workers):
        (values,counts) = np.unique(targets_array, return_counts=True)
        workers_idexes = [[] for _ in range(n_workers)]

        for i,count in zip(values,counts):
            
            indxes = np.arange(len(targets_array))[targets_array == i]
            amount = count//n_workers

            init_pos = 0
            for w in range(n_workers):

                if w%2==0:
                    if i < 3:
                        current_amount = amount + int(amount*0.8)
                    elif i<6:
                        current_amount = amount - int(amount*0.8)
                    elif i<8:
                        current_amount = amount
                    else:
                        current_amount = amount

                else:
                    if i < 3:
                        current_amount = amount - int(amount*0.8)
                    elif i<6:
                        current_amount = amount + int(amount*0.8)
                    elif i<8:
                        current_amount = amount + int(amount*0.6)
                    else:
                        current_amount = amount - int(amount*0.6)
                
                if (w == n_workers-1):
                    current_amount = count - init_pos


                workers_idexes[w] += list(indxes[init_pos:init_pos+current_amount])
                
                init_pos += amount
        
        for w in range(n_workers):
            random.shuffle(workers_idexes[w])

        return workers_idexes
