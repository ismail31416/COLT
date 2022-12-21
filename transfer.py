# Importing Libraries
import argparse
import copy
import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import random
# Custom Libraries
from utils import get_split, compare_models, checkdir
from weights import weight_init
from prune import prune_rate_calculator
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
    
    if args.dataset2 == "mnist":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        traindataset = datasets.MNIST('./data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('./data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2

    elif args.dataset2 == "cifar10":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        traindataset = datasets.CIFAR10('./data', train=True, download=True,transform=train_transformer)
        testdataset = datasets.CIFAR10('./data', train=False, transform=val_transformer)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2 

    elif args.dataset2 == "fashionmnist":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2859,), (0.3530,))])
        traindataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        testdataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2

    elif args.dataset2 == "cifar100":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        traindataset = datasets.CIFAR100('./data', train=True, download=True, transform=train_transformer)
        testdataset = datasets.CIFAR100('./data', train=False, transform=val_transformer)   
        from archs.cifar100 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2 
     
    elif args.dataset2 == "imagenet":
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                             transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                             transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  
            
        traindataset = datasets.ImageNet('./data/imagenet', split='train', transform=train_transform)
        testdataset = datasets.ImageNet('./data/imagenet', split='val', transform=test_transform)   
        from archs.imagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    elif args.dataset2 == "tinyimagenet":
        traindataset, testdataset = get_final_train_and_test_set()
        from archs.tinyimagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2        
    
    else:
        print("\nWrong Dataset choice \n")
        exit()
    
   
    return traindataset, testdataset


def modelselect(args, output_class=None):
     # Importing Network Architecture
     
    if args.dataset1 == "mnist":
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
   
    elif args.dataset1 == "cifar10":
        from archs.cifar10 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    elif args.dataset1 == "fashionmnist":
        from archs.mnist import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
    
    elif args.dataset1 == "cifar100":
        from archs.cifar100 import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2
         
    elif args.dataset1 == "imagenet":
        from archs.imagenet import AlexNet, LeNet5, fc1, resnet, densenet, conv3, mobilenetv2, shufflenetv2

    elif args.dataset1 == "tinyimagenet":
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
            model = AlexNet.AlexNet().to(device)
        else:
            model = AlexNet.AlexNet(num_classes=output_class).to(device)   
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

    train_loader, test_loader = get_split(dataset_name =  args.dataset2, batch_size=args.batch_size, train = traindataset, test = testdataset, model_name = args.part2, shuffle=True)
   
    ## bias always FALSE for Resnet18 as its conv layers has NO bias!!!
    if args.bias == 0:
        bias=False
    else:
        bias=True
    
    if args.resume == 1:
        resume = True
    else:
        resume = False

    comp = []
    bestacc = []
    p_round = 0
    
    if args.prune_type == 'lth':
        transfer_weights = sorted(glob.glob(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset1}/{args.prune_type}/{args.part1}/*.pth"))
    elif args.prune_type == 'colt':
        transfer_weights = sorted(glob.glob(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset1}/{args.prune_type}/*.pth"))        
    
    print(f"The length is: {len(transfer_weights)}")
    
    if resume:
        cp = torch.load(f"{os.getcwd()}/checkpoints/{args.arch_type}/transfers/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/checkpoint_model.pth")
        p_round = cp['round']
        comp = cp['comp']
        bestacc = cp['bestacc']

    # for reproducing random Linear Layer weights
    model0 = modelselect(args, output_class = args.output_class)
            
    weight_dir = f"{os.getcwd()}/saves/{args.arch_type}/transfers/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/output_layer.pth"
    checkdir(f"{os.getcwd()}/saves/{args.arch_type}/transfers/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}")
    
    if args.dataset1 == "cifar10" or args.dataset1 == "fashionmnist" or args.dataset1 == "mnist":
        oc = 10
    elif args.dataset1 == "cifar100":
        oc = 100
    elif args.dataset1 == "tinyimagenet":
        oc = 200
    elif args.dataset1 == "imagenet":
        oc=1000
    
    for i in range(p_round, len(transfer_weights)):
        

        if args.prune_type == 'lth':
    
            model2 = modelselect(args, output_class=oc)
        
        elif args.prune_type == 'colt':
            
            model2 = modelselect(args)
            
        
        transfer_state_dict = torch.load(transfer_weights[i])

        if i!=0:
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

        model2.load_state_dict(transfer_state_dict)
        model2.to(device)
        # print(model2)
        
        # for module2 in model2.modules():
        #     # if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d) or isinstance(module2, nn.BatchNorm2d):
        #     if args.arch_type == "lenet5" or args.arch_type == "fc1":
        #         if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d): 
        #             module2.weight = module2.weight_mask * module2.weight_orig
        #             if bias:
        #                 module2.bias = module2.bias_mask * module2.bias_orig 
        #     else:
        #         if isinstance(module2, nn.Conv2d): # for others
        #             module2.weight = module2.weight_mask * module2.weight_orig


        if i!=0:
            for name2, module2 in model2.named_modules():
                # if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d) or isinstance(module2, nn.BatchNorm2d):
                if args.arch_type == "lenet5" or args.arch_type == "fc1":
                    if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d): 
                        prune.remove(module2, 'weight')
                        if bias: 
                            prune.remove(module2, 'bias')
                else:
                    if isinstance(module2, nn.Conv2d): # for others
                        prune.remove(module2, 'weight')

                    
        # if (args.dataset1 != args.dataset2) and (not (args.dataset1=='cifar10' and args.dataset2=='fashionmnist')) or (not (args.dataset1=='fashionmnist' and args.dataset2=='cifar10')):
        if (args.dataset1 != args.dataset2) or (args.dataset1 == args.dataset2 and args.part2 == "full"):
            if args.arch_type == 'fc1':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    output_state_dict = copy.deepcopy(model0.classifier[4].state_dict())
                    torch.save(output_state_dict, weight_dir)  
                num_ftrs = model2.classifier[4].in_features
                model2.classifier[4] = nn.Linear(num_ftrs, args.output_class).to(device)
                model2.classifier[4].load_state_dict(output_state_dict)
            if args.arch_type == 'lenet5':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    output_state_dict = copy.deepcopy(model0.fc3.state_dict())
                    torch.save(output_state_dict, weight_dir)  
                num_ftrs = model2.fc3.in_features
                model2.fc3 = nn.Linear(num_ftrs, args.output_class).to(device)
                model2.fc3.load_state_dict(output_state_dict)
            elif args.arch_type == 'conv3':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    output_state_dict = copy.deepcopy(model0.fc2.state_dict())
                    torch.save(output_state_dict, weight_dir)
                num_ftrs = model2.fc2.in_features
                model2.fc2 = nn.Linear(num_ftrs, args.output_class).to(device)
                model2.fc2.load_state_dict(output_state_dict)
            elif args.arch_type == 'alexnet':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    output_state_dict = copy.deepcopy(model0.classifier.state_dict())
                    torch.save(output_state_dict, weight_dir) 
                num_ftrs = model2.classifier.in_features
                model2.classifier = nn.Linear(num_ftrs, args.output_class).to(device)
                model2.classifier.load_state_dict(output_state_dict)
            elif args.arch_type == 'resnet18' or args.arch_type == 'densenet121':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    output_state_dict = copy.deepcopy(model0.linear.state_dict())
                    torch.save(output_state_dict, weight_dir) 
                num_ftrs = model2.linear.in_features
                model2.linear = nn.Linear(num_ftrs, args.output_class).to(device)
                model2.linear.load_state_dict(output_state_dict)
            elif args.arch_type == 'mobilenetv2':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    # output_state_dict = copy.deepcopy(model0.conv2.state_dict())
                    output_state_dict = copy.deepcopy(model0.linear.state_dict())
                    torch.save(output_state_dict, weight_dir) 
                # num_ftrs = model2.conv2.in_channels
                num_ftrs = model2.linear.in_features
                # model2.conv2 = nn.Conv2d(num_ftrs, args.output_class, 1).to(device)
                # model2.conv2.load_state_dict(output_state_dict)                
                model2.linear = nn.Linear(num_ftrs, args.output_class).to(device)
                model2.linear.load_state_dict(output_state_dict)                
            elif args.arch_type == 'shufflenetv2':
                if os.path.exists(weight_dir):
                     output_state_dict = torch.load(weight_dir)
                else:
                    output_state_dict = copy.deepcopy(model0.fc.state_dict())
                    torch.save(output_state_dict, weight_dir)  
                num_ftrs = model2.fc.in_features
                model2.fc = nn.Linear(num_ftrs, args.output_class).to(device)  
                model2.fc.load_state_dict(output_state_dict)  

                
        if i==0:
            torch.save(model2.state_dict(), f"{os.getcwd()}/saves/{args.arch_type}/transfers/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/init_for_lth.pth")
            
        # print(model2)
        model2, all_loss2, all_accuracy2, accuracy2, comp2, best_accuracy2, best_val_loss2 = training(model = model2, args = args, train_loader = train_loader, 
                                                                                                        test_loader = test_loader, model_type = args.part2, bias = bias)

        
        comp.append(comp2)
        bestacc.append(best_accuracy2)
    
        p_round = i
        #Saving Cheeckpoint
        checkpoint = {
            'round' : p_round+1,
            'comp' : comp,
            'bestacc' : bestacc,
        }        
                        
        checkdir(f"{os.getcwd()}/checkpoints/{args.arch_type}/transfers/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}")
        torch.save(checkpoint, f"{os.getcwd()}/checkpoints/{args.arch_type}/transfers/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/checkpoint_model.pth")  
        
        # Dump Plot values
        checkdir(f"{os.getcwd()}/dumps/transfers/{args.arch_type}/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}")
        all_loss2.dump(f"{os.getcwd()}/dumps/transfers/{args.arch_type}/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/{args.prune_type}_all_loss_{comp2}.dat")
        all_accuracy2.dump(f"{os.getcwd()}/dumps/transfers/{args.arch_type}/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/{args.prune_type}_all_accuracy_{comp2}.dat")
              


        np_comp = np.array(comp)
        np_bestacc = np.array([x.cpu() for x in bestacc])

        # Dumping Values for Plot
        checkdir(f"{os.getcwd()}/dumps/transfers/{args.arch_type}/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}")
        np_comp.dump(f"{os.getcwd()}/dumps/transfers/{args.arch_type}/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/{args.prune_type}_compression.dat")
        np_bestacc.dump(f"{os.getcwd()}/dumps/transfers/{args.arch_type}/{args.prune_type}/{args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}/{args.prune_type}_bestaccuracy.dat")
        
        if (args.dataset1 == args.dataset2 and args.part2 == "full"):
            checkdir(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset1}/full")
            np_comp.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset1}/full/{args.prune_type}_compression.dat")
            np_bestacc.dump(f"{os.getcwd()}/dumps/{args.prune_type}/{args.arch_type}/{args.dataset1}/full/{args.prune_type}_bestaccuracy.dat")


    # Plotting
    a = np.arange(p_round+1)
    plt.plot(a, np_bestacc, c="blue", label="Winning Tickets") 
    plt.title(f"Test Accuracy VS Unpruned Weights Percentage ({args.prune_type} {args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}, {args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("Test Accuracy") 
    plt.xticks(a, np_comp, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray")
    
    checkdir(f"{os.getcwd()}/plots/transfers/{args.arch_type}/{args.prune_type}")
    plt.savefig(f"{os.getcwd()}/plots/transfers/{args.arch_type}/{args.prune_type}/{args.prune_type} {args.dataset1}{args.part1}_on_{args.dataset2}{args.part2}.png", dpi=1200, bbox_inches='tight') 
    plt.close()   
   
           
    
    
if __name__=="__main__":   
    
    ## Transfer learning from "1" to "2"
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=0, type=int, help="resume training or not")
    parser.add_argument("--augmentations", default="yes", type=str, help="yes | no")#yes
    parser.add_argument("--arch_type", default="mobilenetv2", type=str, help="fc1 | lenet5 | alexnet | conv3 | resnet18 | densenet121 | mobilenetv2 | shufflenetv2")
    parser.add_argument("--dataset1", default="tinyimagenet", type=str, help="mnist | cifar10 | fashionmnist | cifar100 | imagenet | tinyimagenet | -dataset on which the model is already trained")
    parser.add_argument("--dataset2", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100 | imagenet | tinyimagenet | -dataset on which the model will be trained based on weights from dataset1")
    parser.add_argument("--prune_type", default="lth", type=str, help="lth | colt")
    parser.add_argument("--part1", default="full", type=str, help="A | B -whose weights will be transferred") #part1 should be empty string while transferring COLT weights
    parser.add_argument("--part2", default="full", type=str, help="A | B | full -which will be trained on transferred weights") #if "full", lth/colt weights will be transferred to entire dataset  
    parser.add_argument("--bias", default=0, type=int, help="prune bias or not")#0
    parser.add_argument("--output_class", default=10, type=int, help="Output classes for the model (based on dataset2)")
    parser.add_argument("--start_iter", default=0, type=int, help="start epoch")#0
    parser.add_argument("--end_iter", default=50, type=int, help="end epoch")#50
    parser.add_argument("--batch_size", default=256, type=int) #256 
    parser.add_argument("--lr",default= 0.1, type=float, help="Learning rate")#0.01 for AlexNet and 0.1 for all other models
    parser.add_argument("--warmup", default=1, type=int, help="Initial no. of epochs to apply learning rate warmup. 3 means apply warmup for first 3 epochs. 0 means no warmup") #1
    parser.add_argument("--patience", default=50, type=int, help="Patience level of epochs before ending training (based on validation loss)")#50
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