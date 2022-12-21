#ANCHOR Libraries
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from torch.utils.data import Dataset
import random
from torch.optim.lr_scheduler import _LRScheduler


def get_split(dataset_name, batch_size, train, test, model_name, shuffle=False):

    if model_name == 'A' or model_name == 'B':
        if model_name == 'A':
            start_class = 0
            if dataset_name == "cifar10" or dataset_name == "fashionmnist":
                end_class = 5
            elif dataset_name == 'cifar100':
                end_class = 50
            elif dataset_name == 'imagenet':
                end_class = 500
            elif dataset_name == 'tinyimagenet':
                end_class = 100

        elif model_name == 'B':
            if dataset_name == 'cifar10' or dataset_name == 'fashionmnist':
                start_class = 5
                end_class = 10
            elif dataset_name == 'cifar100':
                start_class = 50
                end_class = 100
            elif dataset_name == 'imagenet':
                start_class = 500
                end_class = 1000
            elif dataset_name == 'tinyimagenet':
                start_class = 100
                end_class = 200

        targets_train = torch.tensor(train.targets)
        target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
        target_train_idx = torch.where(target_train_idx==1)[0]    
        target_train_subset = targets_train[target_train_idx]    

        targets_test = torch.tensor(test.targets)
        target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
        target_test_idx = torch.where(target_test_idx==1)[0]    
        target_test_subset = targets_test[target_test_idx]

        if model_name == 'B':
            target_train_subset = target_train_subset - start_class
            target_test_subset = target_test_subset - start_class

        print(f"Max train value is {torch.max(target_train_subset)}, Min train value is {torch.min(target_train_subset)}")
        print(f"Max test value is {torch.max(target_test_subset)}, Min test value is {torch.min(target_test_subset)}")

        print(f"The train target shape is {target_train_subset.size()}")
        print(f"The test target shape is {target_test_subset.size()}")



        class CustomSubset(Dataset):
            r"""
            Subset of a dataset at specified indices.

            Arguments:
                dataset (Dataset): The whole Dataset
                indices (sequence): Indices in the whole set selected for subset
                labels(sequence) : targets as required for the indices. will be the same length as indices
            """
            def __init__(self, dataset, indices, labels):
                self.dataset = torch.utils.data.Subset(dataset, indices)
                self.targets = labels
            def __getitem__(self, idx):
                image = self.dataset[idx][0]
                target = self.targets[idx]
                return (image, target)

            def __len__(self):
                return len(self.targets)


        train_subset = CustomSubset(train, target_train_idx, target_train_subset)
        test_subset = CustomSubset(test, target_test_idx, target_test_subset)


        #pin_memory (bool, optional) – If True, the data loader will copy Tensors into CUDA pinned memory before returning them
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size = batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True) 

    
    elif model_name == 'full':
        
        targets_train = torch.tensor(train.targets)
        targets_test = torch.tensor(test.targets)
        
        print(f"Max train value is {torch.max(targets_train)}, Min train value is {torch.min(targets_train)}")
        print(f"Max test value is {torch.max(targets_test)}, Min test value is {torch.min(targets_test)}")

        print(f"The train target shape is {targets_train.size()}")
        print(f"The test target shape is {targets_test.size()}")
        
        #pin_memory (bool, optional) – If True, the data loader will copy Tensors into CUDA pinned memory before returning them
        train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True) 
    
    train_image, train_label = iter(train_loader).next()
    test_image, test_label = iter(test_loader).next()

    print(f"The train input data shape is: {train_image.shape}")
    print(f"The train label shape is: {train_label.shape}")
    print(f"The test input data shape is: {test_image.shape}")
    print(f"The test label shape is: {test_label.shape}\n")
         
    
    return train_loader, test_loader


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
  

#Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)

'''

def lr_warmup(optimizer,lr):
    """
    Sets the learning rate to a particular value
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer
    '''
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        '''
        Since step() should be invoked after each batch instead of after each epoch, this number(last_epoch) represents the total number of batches computed, not the total number of epochs computed. 
        When last_epoch=-1, the scheduler is started from the beginning. Else, last epoch is first 1 since 1st batch is computed, then 2, then 3, and so on until total_iters(no. of batches/iterations).
        '''
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]