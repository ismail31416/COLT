from torch.utils.data import Dataset, DataLoader
from skimage import io
import cv2
import glob
import random
import torch
import os
import torchvision.transforms as transforms
import numpy as np
                   
####################################################
#       Create Train and Val sets
####################################################
def flatten(t):
    return [item for sublist in t for item in sublist]


def generate_train_val_image_path():
    train_data_path = './data/tiny-imagenet-200/train'
    val_data_path = './data/tiny-imagenet-200/val'

    train_image_paths = []  # to store image paths in list
    classes = []  # to store class values

    for data_path in glob.glob(train_data_path + '/*'):
#         print(data_path)
#         break
        classes.append(data_path.split('/')[-1])
        train_image_paths.append(glob.glob(data_path + '/*'))

    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)
    train_image_paths = [path.replace('\\', '/') for path in train_image_paths]
    
    print('train_image_path example: ', train_image_paths[0])
    print('class example: ', classes[0])


    val_image_paths = []
    for data_path in glob.glob(val_data_path + '/*'):
        val_image_paths.append(glob.glob(data_path + '/*'))

    val_image_paths = list(flatten(val_image_paths))
    val_image_paths = [path.replace('\\', '/') for path in val_image_paths]
    

    print("Train size: {}\nValid size: {}\n".format(len(train_image_paths), len(val_image_paths)))

    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}

    return train_image_paths, val_image_paths, idx_to_class, class_to_idx, classes




#######################################################
#               Define Dataset Class
#######################################################

class TinyImagenetDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform = None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.targets =  [self.class_to_idx[path.split("/")[-2]] for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_filepath = self.image_paths[index]
        image_filepath = image_filepath.replace('\\', '/')
        image = io.imread(image_filepath) # your slow data loading
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
                
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_final_train_and_test_set():
    
    train_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(20), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
                          transforms.Normalize(mean = [0.4802, 0.4481, 0.3975], std = [0.2302, 0.2265, 0.2262])])

    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 
                          transforms.Normalize(mean = [0.4802, 0.4481, 0.3975], std = [0.2302, 0.2265, 0.2262])])
    
    train_image_paths, val_image_paths, idx_to_class, class_to_idx, classes = generate_train_val_image_path()
    trainset = TinyImagenetDataset(train_image_paths, class_to_idx, transform = train_transform)
    testset = TinyImagenetDataset(val_image_paths, class_to_idx, transform = test_transform)


    return trainset, testset