import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm


DPI = 1200
# prune_iterations = 35
datasets = ["cifar10"]
parts = ["full"]
arch_types = ["mobilenetv2"]
typ = ["lth", "colt"]

for dataset in tqdm(datasets):
    
    for part in tqdm(parts):
        
        min_y_limit = []
        max_y_limit = []
        for arch_type in tqdm(arch_types):
            
            ## haven't trained tinyimagenet on conv6!!!!
            if arch_type == "conv6" and dataset == "tinyimagenet":
                continue
            
            for t in typ:
        
                if t == "lth":
                    d = np.load(f"{os.getcwd()}/dumps/{t}/{arch_type}/{dataset}/{part}/{t}_compression.dat", allow_pickle=True)
                    d = 100 - d
                    b1 = np.load(f"{os.getcwd()}/dumps/{t}/{arch_type}/{dataset}/{part}/{t}_bestaccuracy.dat", allow_pickle=True)
                    length = len(glob.glob(f"./saves/{arch_type}/{dataset}/{t}/{part}/*.pth"))
                    print(f"lth is {length+1}")
                    plt.plot(d, b1, label=f"{t}-{arch_type}") 
                    
                    # to save as csv
                    acc = np.around(b1, decimals=2)
                    rate = np.around(d, decimals=2)
                    data = {"rate":rate, "accuracy":acc}
                    df = pd.DataFrame(data, columns=["rate", "accuracy"])
                    df.to_csv(f"./result-csvs/{dataset}{part}_{t}_{arch_type}.csv", index=False)
                    
                if t == "colt":
                    d = np.load(f"{os.getcwd()}/dumps/{t}/{arch_type}/{dataset}/{part}/{t}_compression.dat", allow_pickle=True)
                    d = 100 - d
                    b2 = np.load(f"{os.getcwd()}/dumps/{t}/{arch_type}/{dataset}/{part}/{t}_bestaccuracy.dat", allow_pickle=True)
                    length = len(glob.glob(f"./saves/{arch_type}/{dataset}/{t}/*.pth"))
                    print(f"colt is {length+1}")
                    plt.plot(d, b2, label=f"{t}-{arch_type}") 
                    
                    # to save as csv
                    acc = np.around(b2, decimals=2)
                    rate = np.around(d, decimals=2)
                    data = {"rate":rate, "accuracy":acc}
                    df = pd.DataFrame(data, columns=["rate", "accuracy"])
                    df.to_csv(f"./result-csvs/{dataset}{part}_{t}_{arch_type}.csv", index=False)

                #plt.clf()
                #sns.set_style('darkgrid')
                #plt.style.use('seaborn-darkgrid')
                # a = np.arange(length+1)
                # plt.plot(d, b, label=f"{t}-{arch_type}") 
                # plt.plot(a, c, c="red", label="Random reinit") 


            plt.title(f"{dataset}-{part}") 
            plt.xlabel("Prune Rate (%)") 
            plt.ylabel("Test Accuracy (%)") 

            min_y = min(min(b1), min(b2))
            max_y = max(max(b1), max(b2))
            plt.ylim(min_y, max_y+1)
            
            min_y_limit.append(min_y)
            max_y_limit.append(max_y)
            
            # plt.xticks(a, d, rotation ="vertical")                        
            # plt.yticks(np.arange(min(min(b1), min(b2)), max(max(b1), max(b2))+1, 1.0))
            plt.legend() 
            plt.grid(color="gray") 
            
            # uncomment this block to plot modelwise transfer plots for each dataset and comment the block below this block
#             directory = f"{os.getcwd()}/plots/combined_plots/"
#             if not os.path.exists(directory):
#                 os.makedirs(directory)

#             plt.savefig(directory+f"{arch_type}_{dataset}_{part}.png", dpi=DPI, bbox_inches='tight') 
#             plt.close()
#             print(f"\n combined_{arch_type}_{dataset}_{part} plotted!\n")

#         #uncomment this block to plot all models datasetwise transfer plots and comment the upper block
        directory = f"{os.getcwd()}/plots/combined_plots/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.ylim(min(min_y_limit), max(max_y_limit)+1)
        plt.savefig(directory+f"all_{dataset}_{part}.png", dpi=DPI, bbox_inches='tight') 
        plt.close()
        print(f"\n all_{dataset}_{part} plotted!\n")
        
        

# DPI = 1200
# prune_iterations = 35
# arch_types = ["fc1", "lenet5", "resnet18"]
# datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]


# for arch_type in tqdm(arch_types):
#     for dataset in tqdm(datasets):
#         d = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_compression.dat", allow_pickle=True)
#         b = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_bestaccuracy.dat", allow_pickle=True)
#         c = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/reinit_bestaccuracy.dat", allow_pickle=True)

#         #plt.clf()
#         #sns.set_style('darkgrid')
#         #plt.style.use('seaborn-darkgrid')
#         a = np.arange(prune_iterations)
#         plt.plot(a, b, c="blue", label="Winning tickets") 
#         plt.plot(a, c, c="red", label="Random reinit") 
#         plt.title(f"Test Accuracy vs Weights % ({arch_type} | {dataset})") 
#         plt.xlabel("Weights %") 
#         plt.ylabel("Test accuracy") 
#         plt.xticks(a, d, rotation ="vertical") 
#         plt.ylim(0,100)
#         plt.legend() 
#         plt.grid(color="gray") 

#         plt.savefig(f"{os.getcwd()}/plots/lt/combined_plots/combined_{arch_type}_{dataset}.png", dpi=DPI, bbox_inches='tight') 
#         plt.close()
#         #print(f"\n combined_{arch_type}_{dataset} plotted!\n")