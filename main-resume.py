## NEEDS TO BE FIXED OR CHECKED!!!!!!!!

# Importing Libraries
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
# import torchvision.utils as vutils
import seaborn as sns
import pickle
import random
# Custom Libraries
from utils import get_split, compare_models, plot_train_test_stats, checkdir
from weights import weight_init, weight_rewinding
from prune import colt, prune_rate_calculator, lth_local_unstructured_pruning, lth_global_unstructured_pruning
from trainresume import training, train, test



# CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure Reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# Tensorboard initialization
writer = SummaryWriter()
# Plotting Style
sns.set_style('darkgrid')

        
def dataset(args):
  # Data Loader
    if args.dataset == "mnist":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        traindataset = datasets.MNIST('./data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('./data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6

    elif args.dataset == "cifar10":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        traindataset = datasets.CIFAR10('./data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('./data', train=False, transform=transform)      
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6 

    elif args.dataset == "fashionmnist":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2859,), (0.3530,))])
        traindataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        testdataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6

    elif args.dataset == "cifar100":
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        traindataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR100('./data', train=False, transform=transform)   
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet, densenet, conv6  
     
    elif args.dataset == "imagenet":
        transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        traindataset = datasets.ImageNet('./data/imagenet', split='train', transform=transform)
        testdataset = datasets.ImageNet('./data/imagenet', split='val', transform=transform)   
        from archs.imagenet import AlexNet, fc1, LeNet5, vgg, resnet, densenet, conv6 
        
    else:
        print("\nWrong Dataset choice \n")
        exit()
    
   
    return traindataset,testdataset


def modelselect(args):
     # Importing Network Architecture
     
    if args.dataset == "mnist":
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6
    elif args.dataset == "cifar10":
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6
    
    elif args.dataset == "fashionmnist":
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6
    
    elif args.dataset == "cifar100":
        from archs.cifar100 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6  
         
    elif args.dataset == "imagenet":
        from archs.imagenet import AlexNet, LeNet5, fc1, vgg, resnet, densenet, conv6  
    
    if args.arch_type == "fc1":
        model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16_bn().to(device)  
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)   
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)   
    elif args.arch_type == 'conv6':
        model = conv6.conv6().to(device)
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

    train_loaderA, test_loaderA = get_split(dataset_name =  args.dataset, batch_size=args.batch_size,train = traindataset, test = testdataset, model_name = 'A', shuffle=True)
    train_loaderB, test_loaderB = get_split(dataset_name =  args.dataset, batch_size=args.batch_size,train = traindataset, test = testdataset, model_name = 'B', shuffle=False)
    
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
    else:
        bias=True
        
    if args.resume == 0:
        resume=False
    else:
        resume=True

    compa = []
    bestacc1 = []
    compb = []
    bestacc2 = []
    
    if pt == 'lth':
        #Initiqlize a model with random weights
        model0 = modelselect(args)
        model0.apply(weight_init)

        # Copying and Saving Initial State
        initial_state_dict = copy.deepcopy(model0.state_dict())
        checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
        torch.save(model0, f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict.pth.tar")
        torch.save(model0.state_dict(), f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict.pth")             
    
    elif pt == 'colt':
        model0 = modelselect(args)
        
        initial_state_dict = torch.load(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict.pth")
        model0.load_state_dict(initial_state_dict)

                 
    
    model1 = modelselect(args)
    model1.apply(weight_init) 
    model1.load_state_dict(initial_state_dict)
    
    model2 = modelselect(args)
    model2.apply(weight_init)
    model2.load_state_dict(initial_state_dict)
    
    
    compare_models(model1,model2)
    
    #model1.load_state_dict(model0.state_dict())
    #model2.load_state_dict(model0.state_dict())

    while (prune_rate < args.final_prune_rate):
                
        print(f"\n--- Pruning Level [{ITE}: Pruning Round: {pruning_round}]")
        
        if pruning_round == 0:

            model1, all_loss1, all_accuracy1, accuracy1, comp1, best_accuracy1, best_val_loss1 = training(model=model1, args = args, train_loader=train_loaderA, test_loader = test_loaderA, model_type="A", bias=bias,                                                                                                                                     pruning_round=pruning_round, prune_rate=prune_rate, compa=compa, compb=compb, bestacc1=bestacc1, bestacc2=bestacc2)
            model2, all_loss2, all_accuracy2, accuracy2, comp2, best_accuracy2, best_val_loss2 = training(model=model2, args = args, train_loader=train_loaderB, test_loader = test_loaderB, model_type="B", bias=bias,                                                                                                                                     pruning_round=pruning_round, prune_rate=prune_rate, compa=compa, compb=compb, bestacc1=bestacc1, bestacc2=bestacc2)
            

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
                    
            
            model1, all_loss1, all_accuracy1, accuracy1, comp1, best_accuracy1, best_val_loss1 = training(model=model1, args = args, train_loader=train_loaderA, test_loader = test_loaderA, model_type="A", bias=bias,                                                                                                                                     pruning_round=pruning_round, prune_rate=prune_rate, compa=compa, compb=compb, bestacc1=bestacc1, bestacc2=bestacc2)
            
            model2, all_loss2, all_accuracy2, accuracy2, comp2, best_accuracy2, best_val_loss2 = training(model=model2, args = args, train_loader=train_loaderB, test_loader = test_loaderB, model_type="B", bias=bias,                                                                                                                                     pruning_round=pruning_round, prune_rate=prune_rate, compa=compa, compb=compb, bestacc1=bestacc1, bestacc2=bestacc2)
        
        compa.append(comp1)
        compb.append(comp2)
        
        writer.add_scalar('Accuracy/test', best_accuracy1, comp1)
        bestacc1.append(best_accuracy1)
        

        writer.add_scalar('Accuracy/test', best_accuracy2, comp2)
        bestacc2.append(best_accuracy2)
        
        
        pruning_round += 1


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


    compa = np.array(compa)
    bestacc1 = np.array([x.item() for x in bestacc1])
    compb = np.array(compb)
    bestacc2 = np.array([x.item() for x in bestacc2])
    
    # Dumping Values for Plotcheckdir(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}checkdir(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B")
    compa.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_compression.dat")
    bestacc1.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_bestaccuracy.dat")

    compb.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_compression.dat")
    bestacc2.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_bestaccuracy.dat")

    # Plotting
    a = np.arange(pruning_round)
    plt.plot(a, bestacc1, c="blue", label="Winning Tickets") 
    plt.title(f"Test Accuracy VS Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("Test Accuracy") 
    plt.xticks(a, compa, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray")
    
    checkdir(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/A")
    plt.savefig(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/A/{pt}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()   

    # Plotting
    # a = np.arange(pruning_round)
    plt.plot(a, bestacc2, c="blue", label="Winning Tickets") 
    plt.title(f"Test Accuracy VS Unpruned Weights Percentage ({args.dataset},{args.arch_type})") 
    plt.xlabel("Unpruned Weights Percentage") 
    plt.ylabel("Test Accuracy") 
    plt.xticks(a, compb, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray")
    
    checkdir(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/B")
    plt.savefig(f"{os.getcwd()}/plots/{pt}/{args.arch_type}/{args.dataset}/B/{args.prune_type}_AccuracyVsWeights.png", dpi=1200) 
    plt.close()      
           
    
    
if __name__=="__main__":   
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default = 0, type=int, help="Resume training from a checkpoint or not")
    parser.add_argument("--lr", default= 0.001, type=float, help="Learning rate") #1.0e-3
    parser.add_argument("--batch_size", default=256, type=int) #256 for conv6, alexnet
    parser.add_argument("--start_iter", default=0, type=int, help="start epoch")
    parser.add_argument("--end_iter", default=100, type=int, help="end epoch")
    parser.add_argument("--prune_type", default="lth", type=str, help="lth | colt")
    parser.add_argument("--prune_strategy", default="local", type=str, help="global | local")
    parser.add_argument("--bias", default="1", type=int, help="prune bias or not")
    parser.add_argument("--dataset", default="cifar10", type=str, help="mnist | cifar10 | fashionmnist | cifar100 | imagenet")
    parser.add_argument("--arch_type", default="conv6", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--output_class", default=5, type=int, help="Output classes for the model")
    parser.add_argument("--linear_prune_ratio", default=0.20, type=float, help="Linear layer pruning proportion")
    parser.add_argument("--output_prune_ratio", default=0.10, type=float, help="Output layer pruning proportion")
    parser.add_argument("--conv_prune_ratio", default=0.15, type=float, help="Conv layer pruning proportion")
    parser.add_argument("--batchnorm_prune_ratio", default=0.0, type=float, help="Batchnorm layer pruning proportion")
    parser.add_argument("--final_prune_rate", default=10, type=float, help="Final prune rate before pruning stops")
    parser.add_argument("--patience", default=20, type=int, help="Patience level of epochs before ending training (based on validation loss)")
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--gpu", default="0", type=str)

    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    main(args, ITE=1)
       
    # Looping Entire process
    # for i in range(0, 5):
    #     main(args, ITE=i)
    
    
    
