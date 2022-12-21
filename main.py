# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
# import torchvision.utils as vutils
import seaborn as sns
import pickle
import random
# Custom Libraries
from utils import get_split, compare_models, checkdir
from weights import weight_init, weight_rewinding
from prune import colt, prune_rate_calculator, lth_local_unstructured_pruning, lth_global_unstructured_pruning
from train import training
from tinyimagenet import get_final_train_and_test_set


# CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# Ensure Reproducibility
# torch.use_deterministic_algorithms(True, warn_only=True)   
# torch.backends.cudnn.enabled = False  ## doubles training time!!!!
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

# Plotting Style
sns.set_style('darkgrid')

        
def dataset(args):
    # Data Loader
    # For transfer learning from RGB(Cifar10, Cifar100) to GreyScale(MNIST) use transformation "Grayscale(3) which will create 2 mre copies of the channel where R==G==B"
    # Details: https://pytorch.org/vision/main/generated/torchvision.transforms.Grayscale.html 
    #https://discuss.pytorch.org/t/transfer-learning-for-images-of-single-channels/24228/2
    if args.augmentations == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    # transformer for val set
    val_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    
    if args.dataset == "mnist":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        traindataset = datasets.MNIST('./data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('./data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2

    elif args.dataset == "cifar10":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        traindataset = datasets.CIFAR10('./data', train=True, download=True,transform=train_transformer)
        testdataset = datasets.CIFAR10('./data', train=False, transform=val_transformer)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2 

    elif args.dataset == "fashionmnist":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2859,), (0.3530,))])
        traindataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        testdataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2

    elif args.dataset == "cifar100":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        traindataset = datasets.CIFAR100('./data', train=True, download=True, transform=train_transformer)
        testdataset = datasets.CIFAR100('./data', train=False, transform=val_transformer)   
        from archs.cifar100 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2 
     
    elif args.dataset == "imagenet":
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                             transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                             transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  
            
        traindataset = datasets.ImageNet('./data/imagenet', split='train', transform=train_transform)
        testdataset = datasets.ImageNet('./data/imagenet', split='val', transform=test_transform)   
        from archs.imagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    elif args.dataset == "tinyimagenet":
        traindataset, testdataset = get_final_train_and_test_set()
        from archs.tinyimagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2        
    
    else:
        print("\nWrong Dataset choice \n")
        exit()
    
   
    return traindataset, testdataset


def modelselect(args, output_class=None):
     # Importing Network Architecture
     
    if args.dataset == "mnist":
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
   
    elif args.dataset == "cifar10":
        from archs.cifar10 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    elif args.dataset == "fashionmnist":
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    elif args.dataset == "cifar100":
        from archs.cifar100 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
         
    elif args.dataset == "imagenet":
        from archs.imagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2

    elif args.dataset == "tinyimagenet":
        from archs.tinyimagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    if args.arch_type == "fc1":
        if output_class is None:
            model = fc1.fc1().to(device)
        else:
            model = fc1.fc1(num_classes=output_class).to(device)
    elif args.arch_type == "lenet5":
        if output_class is None:
            model = LeNet5.LeNet5().to(device)
        else:
            model = LeNet5.LeNet5(num_classes=output_class).to(device)
    elif args.arch_type == "alexnet":
        if output_class is None:
            model = AlexNet.alexnet().to(device)
        else:
            model = AlexNet.alexnet(num_classes=output_class).to(device)   
    # elif args.arch_type == "vgg16":
    #     if output_classis None:
    #         model = vgg.vgg16_bn().to(device)
    #     else:
    #         model = vgg.vgg16_bn(num_classes=output_class).to(device)
    elif args.arch_type == "resnet18":
        if output_class is None:
            model = resnet.resnet18().to(device)
        else:
            model = resnet.resnet18(num_classes=output_class).to(device)
    elif args.arch_type == "densenet121":
        if output_class is None:
            model = densenet.densenet121().to(device) 
        else:
            model = densenet.densenet121(num_classes=output_class).to(device) 
    elif args.arch_type == 'conv3':
        if output_class is None:
            model = conv3.conv3().to(device)
        else:
            model = conv3.conv3(num_classes=output_class).to(device)
    elif args.arch_type == "mobilenetv2":
        if output_class is None:
            model = mobilenetv2.mobilenetv2().to(device)
        else:
            model = mobilenetv2.mobilenetv2(num_classes=output_class).to(device)
    elif args.arch_type == "shufflenetv2":
        if output_class is None:
            model = shufflenetv2.shufflenetv2().to(device)
        else:
            model = shufflenetv2.shufflenetv2(num_classes=output_class).to(device)
    else:
        print("\nWrong Model choice\n")
        exit()
    
    return model



#MAIN
def main(args, ITE=1):
    
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
    else:
        print("\n\n ---- CUDA NOT AVAILABLE !!! ---- \n\n")

    traindataset, testdataset = dataset(args)

    train_loaderA, test_loaderA = get_split(dataset_name =  args.dataset, batch_size=args.batch_size, train = traindataset, test = testdataset, model_name = 'A', shuffle=True)
    train_loaderB, test_loaderB = get_split(dataset_name =  args.dataset, batch_size=args.batch_size, train = traindataset, test = testdataset, model_name = 'B', shuffle=True)
    
    pt = args.prune_type
    strategy = args.prune_strategy
    output_class = args.output_class
    prune_rate = 0.0
    pruning_round = 0
    linear_ratio = args.linear_prune_ratio
    output_ratio = args.output_prune_ratio
    conv_ratio = args.conv_prune_ratio
    batchnorm_ratio = args.batchnorm_prune_ratio
    
    ## bias always FALSE for Resnet18 as its conv layers has NO bias!!!
    if args.bias == 0:
        bias=False
    elif args.bias == 1:
        bias=True
    
    if args.resume == 1:
        resume = True
    elif args.resume == 0:
        resume = False

    compa = []
    bestacc1 = []
    compb = []
    bestacc2 = []
    
    weight_dir = f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict.pth"
    
    ## required for weight rewinding
    model0 = modelselect(args) ## initialize a model with random weights
    if os.path.exists(weight_dir):
        initial_state_dict = torch.load(weight_dir)
        model0.load_state_dict(initial_state_dict)
    else:
        # Copying and Saving Initial State
        initial_state_dict = copy.deepcopy(model0.state_dict())
        checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
        torch.save(model0.state_dict(), weight_dir)      
                

    if not resume:             
        model1 = modelselect(args)
        model1.load_state_dict(initial_state_dict)
        model1.to(device)

        model2 = modelselect(args)
        model2.load_state_dict(initial_state_dict)
        model2.to(device)

        compare_models(model1,model2)
    
    else:
        model1 = modelselect(args)
        
        model2 = modelselect(args)

        cp = torch.load(f"{os.getcwd()}/checkpoints/{args.arch_type}/{args.dataset}/{args.prune_type}/checkpoint_model_A_B.pth")
        
        pruning_round = cp['pruning_round']
        prune_rate = cp['prune_rate']
        compa = cp['compa']
        bestacc1 = cp['bestacc1']
        compb = cp['compb']
        bestacc2 = cp['bestacc2']
        
        if pruning_round==1:
            model1.load_state_dict(cp['model_state_dict1'])
            model2.load_state_dict(cp['model_state_dict2'])
            model1.to(device)
            model2.to(device)
        else:                        
            for module1 in model1.modules():
                # if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d) or isinstance(module1, nn.BatchNorm2d):
                if args.arch_type == "lenet5" or args.arch_type == "fc1":
                    if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d): # for others
                        prune.identity(module1, 'weight')
                        if bias:
                            prune.identity(module1, 'bias')
                else:
                    if isinstance(module1, nn.Conv2d):
                        prune.identity(module1, 'weight')     
                        
            for module2 in model2.modules():
                # if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d) or isinstance(module2, nn.BatchNorm2d):
                if args.arch_type == "lenet5" or args.arch_type == "fc1":
                    if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d): # for others
                        prune.identity(module2, 'weight')
                        if bias:
                            prune.identity(module2, 'bias') 
                else: 
                    if isinstance(module2, nn.Conv2d): 
                        prune.identity(module2, 'weight')     
                                              
            model1.load_state_dict(cp['model_state_dict1'])
            model2.load_state_dict(cp['model_state_dict2'])
            model1.to(device)
            model2.to(device)
            
            for module1 in model1.modules():
                # if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d) or isinstance(module1, nn.BatchNorm2d):
                if args.arch_type == "lenet5" or args.arch_type == "fc1":
                    if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d): 
                        module1.weight = module1.weight_mask * module1.weight_orig
                        if bias:
                            module1.bias = module1.bias_mask * module1.bias_orig
                else:
                    if isinstance(module1, nn.Conv2d): 
                        module1.weight = module1.weight_mask * module1.weight_orig    
                        
            for module2 in model2.modules():
                # if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d) or isinstance(module2, nn.BatchNorm2d):
                if args.arch_type == "lenet5" or args.arch_type == "fc1":
                    if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d): 
                        module2.weight = module2.weight_mask * module2.weight_orig
                        if bias:
                            module2.bias = module2.bias_mask * module2.bias_orig  
                else:
                    if isinstance(module2, nn.Conv2d): 
                        module2.weight = module2.weight_mask * module2.weight_orig
                        
                        
    while (prune_rate < args.final_prune_rate):
                
        print(f"\n--- Pruning Level [{ITE}: Pruning Round: {pruning_round}]")
        
        if pruning_round == 0:
        
            if pt == 'colt':
                prune_rate = prune_rate_calculator(model1, bias=bias)
                
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}")
                torch.save(model1.state_dict(),f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/{prune_rate:.2f}_model_{pt}.pth")
                                    
            elif pt == 'lth':           
                prune_rate = prune_rate_calculator(model1, bias=bias)
                
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/A")
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B")
                torch.save(model1.state_dict(),f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/A/{prune_rate:.2f}_model_{pt}.pth")
                torch.save(model2.state_dict(),f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B/{prune_rate:.2f}_model_{pt}.pth")  


            model1, all_loss1, all_accuracy1, accuracy1, comp1, best_accuracy1, best_val_loss1 = training(model=model1, args = args, train_loader=train_loaderA, 
                                                                                                    test_loader = test_loaderA, model_type="A", bias=bias)
            model2, all_loss2, all_accuracy2, accuracy2, comp2, best_accuracy2, best_val_loss2 = training(model=model2, args = args, train_loader=train_loaderB, 
                                                                                                    test_loader = test_loaderB, model_type="B", bias=bias)          

        if not pruning_round == 0:
            
            if strategy == 'local':
                model1 = lth_local_unstructured_pruning(model1, output_class, bias=bias, conv=conv_ratio, linear=linear_ratio, output=output_ratio, batchnorm=batchnorm_ratio)
                model2 = lth_local_unstructured_pruning(model2, output_class, bias=bias, conv=conv_ratio, linear=linear_ratio, output=output_ratio, batchnorm=batchnorm_ratio)
                
            elif strategy == 'global':
                model1 = lth_global_unstructured_pruning(model1, output_class, bias=bias, conv=conv_ratio, linear=linear_ratio, output=output_ratio, batchnorm=batchnorm_ratio)
                model2 = lth_global_unstructured_pruning(model2, output_class, bias=bias, conv=conv_ratio, linear=linear_ratio, output=output_ratio, batchnorm=batchnorm_ratio)
                            
            
            model1 = weight_rewinding(model1, model0, bias=bias)
            model2 = weight_rewinding(model2, model0, bias=bias)

            if pt == 'colt':
                ## keep overlapping weights, make zero non-overlapping ones
                model1,model2 = colt(model1=model1,model2=model2, bias=bias)
                compare_models(model1, model2)

                prune_rate = prune_rate_calculator(model1, bias=bias)
                
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}")
                torch.save(model1.state_dict(),f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/{prune_rate:.2f}_model_{pt}.pth")
                    
                
            elif pt == 'lth':           
                prune_rate = prune_rate_calculator(model1, bias=bias)
                
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/A")
                checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B")
                torch.save(model1.state_dict(),f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/A/{prune_rate:.2f}_model_{pt}.pth")
                torch.save(model2.state_dict(),f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B/{prune_rate:.2f}_model_{pt}.pth")
            
            
            model1, all_loss1, all_accuracy1, accuracy1, comp1, best_accuracy1, best_val_loss1 = training(model=model1, args = args, train_loader=train_loaderA, 
                                                                                                    test_loader = test_loaderA, model_type="A", bias=bias)
            
            model2, all_loss2, all_accuracy2, accuracy2, comp2, best_accuracy2, best_val_loss2 = training(model=model2, args = args, train_loader=train_loaderB, 
                                                                                                    test_loader = test_loaderB, model_type="B", bias=bias)
        
        compa.append(comp1)
        compb.append(comp2)
        
        bestacc1.append(best_accuracy1)
        bestacc2.append(best_accuracy2)
        
                
        pruning_round += 1

        #Saving Cheeckpoint
        checkpoint = {
            'pruning_round' : pruning_round,
            'prune_rate' : prune_rate,
            'compa' : compa,
            'compb' : compb,
            'bestacc1' : bestacc1,
            'bestacc2' : bestacc2,
            'model_state_dict1' : model1.state_dict(),
            'model_state_dict2' : model2.state_dict(),

        }        
        checkdir(f"{os.getcwd()}/checkpoints/{args.arch_type}/{args.dataset}/{args.prune_type}")
        torch.save(checkpoint, f"{os.getcwd()}/checkpoints/{args.arch_type}/{args.dataset}/{args.prune_type}/checkpoint_model_A_B.pth")  
        
        # Dump Plot values
        checkdir(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A")
        checkdir(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B")
        all_loss1.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_all_loss_{comp1}.dat")
        all_accuracy1.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_all_accuracy_{comp1}.dat")
        all_loss2.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_all_loss_{comp2}.dat")
        all_accuracy2.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_all_accuracy_{comp2}.dat")

        '''
        # Dumping mask
        checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
            pickle.dump(mask, fp)'''
              


        np_compa = np.array(compa)
        np_bestacc1 = np.array([x.cpu() for x in bestacc1])
        np_compb = np.array(compb)
        np_bestacc2 = np.array([x.cpu() for x in bestacc2])

        # Dumping Values for Plot
        np_compa.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_compression.dat")
        np_bestacc1.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_bestaccuracy.dat")

        np_compb.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_compression.dat")
        np_bestacc2.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_bestaccuracy.dat")

    # Plotting
    a = np.arange(pruning_round)
    plt.plot(a, np_bestacc1, c="blue", label="Winning Tickets") 
    plt.title(f"Test Accuracy VS Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("Test Accuracy") 
    plt.xticks(a, compa, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray")
    
    checkdir(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/A")
    plt.savefig(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_AccuracyVsWeights.png", dpi=1200, bbox_inches='tight') 
    plt.close()   

    # Plotting
    # a = np.arange(pruning_round)
    plt.plot(a, np_bestacc2, c="blue", label="Winning Tickets") 
    plt.title(f"Test Accuracy VS Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("Test Accuracy") 
    plt.xticks(a, compb, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray")
    
    checkdir(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/B")
    plt.savefig(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/B/{args.prune_type}_AccuracyVsWeights.png", dpi=1200, bbox_inches='tight') 
    plt.close()      
           
    
    
if __name__=="__main__":   
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=0, type=int, help="resume training or not")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate") #0.01 for AlexNet and 0.1 for all other models
    parser.add_argument("--warmup", default=1, type=int, help="1 means to apply warmup to first epoch, 0 means no wamrup for first epoch") #1
    parser.add_argument("--batch_size", default=256, type=int) #256 
    parser.add_argument("--start_iter", default=0, type=int, help="start epoch") #0
    parser.add_argument("--end_iter", default=50, type=int, help="end epoch") #50
    parser.add_argument("--prune_type", default="colt", type=str, help="lth | colt")
    parser.add_argument("--augmentations", default="yes", type=str, help="yes | no") #yes
    parser.add_argument("--prune_strategy", default="global", type=str, help="global | local")
    parser.add_argument("--bias", default=0, type=int, help="prune bias or not") #0
    parser.add_argument("--dataset", default= "cifar100", type=str, help="mnist | cifar10 | fashionmnist | cifar100 | imagenet | tinyimagenet")
    parser.add_argument("--arch_type", default="conv3", type=str, help="fc1 | lenet5 | alexnet | conv3 | resnet18 | densenet121 | mobilenetv2 | shufflenetv2")
    parser.add_argument("--output_class", default=50, type=int, help="Output classes for the model")
    parser.add_argument("--linear_prune_ratio", default=0.0, type=float, help="Linear layer pruning proportion")
    parser.add_argument("--output_prune_ratio", default=0.0, type=float, help="Output layer pruning proportion")
    parser.add_argument("--conv_prune_ratio", default=0.15, type=float, help="Conv layer pruning proportion")
    parser.add_argument("--batchnorm_prune_ratio", default=0.0, type=float, help="Batchnorm layer pruning proportion")
    parser.add_argument("--final_prune_rate", default=99.1, type=float, help="Final prune rate before pruning stops") #99.1
    parser.add_argument("--patience", default=50, type=int, help="Patience level of epochs before ending training (based on validation loss)") #50
    parser.add_argument("--print_freq", default=1, type=int)#1
    parser.add_argument("--valid_freq", default=1, type=int)#1
    parser.add_argument("--gpu", default="0", type=str)#0

    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    main(args, ITE=1)
       
    # Looping Entire process
    # for i in range(0, 5):
    #     main(args, ITE=i)
    
    
    
