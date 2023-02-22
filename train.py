import numpy as np
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from weights import print_nonzeros
from utils import WarmUpLR  

import random
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def training(model, args, train_loader, test_loader, model_type, bias):
    
    print('----------------------')
    print('model '+model_type)
    
    print('----------------------')
    
    
    best_accuracy = 0.0
    patience_count = 0
    best_val_loss = float('inf')
    accuracy=0.0
    bestmodel = model
    all_loss = np.zeros(args.end_iter,float)
    all_accuracy = np.zeros(args.end_iter,float)
    
    if args.arch_type == 'conv3' or args.arch_type == 'fc1' or args.arch_type == 'lenet5':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # no. of batches/iterations per epoch
    iter_per_epoch = len(train_loader)   ## required for warmup
    
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup)
    
    # milestones start counting from 1. so epoch 1,2,3... NOT 0,1,2,3.. so 20 in milestone means 19th epoch for us as we start from epoch 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,45], gamma=0.2) # gamma 0.2 means decrease lr by 5 times (1/5)=0.2 (1/2.5)=0.4
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # ReduceLROnPlateau scheduler reads a metrics quantity and if no improvement is seen for a patience number of epochs, the learning rate is reduced.
    # In min mode, lr will be reduced when the metric has stopped decreasing usually val_loss
    # In max mode, lr will be reduced when the metric has stopped increasing  usually val_acc
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor=0.2, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 20, factor=0.2, verbose=True)
    
    criterion = nn.CrossEntropyLoss() # Default was F.nll_loss

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())
    # Print the table of Nonzeros in each layer
    comp1 = print_nonzeros(model, bias=bias)
    #comp[ite] = comp1 retun comp1de
    pbar = tqdm(range(args.end_iter))
    patience_count = 0
    for iter_ in pbar:
            
            if patience_count >= args.patience:
                print("\n'EarlyStopping' called!\n")
                break
            
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy, test_loss = test(model, test_loader, criterion)
                
                # Check Early Stopping with respect to val loss
                if test_loss < best_val_loss:
                    # update 'best_val_loss' variable to lowest loss encountered so far-
                    best_val_loss = test_loss
                    # reset 'patience_count' variable-
                    patience_count = 0
                # there is no improvement in monitored metric 'val_loss'    
                else:  
                    # number of epochs without any improvement
                    patience_count += 1
                
                # # Check Early Stopping with respect to val accuracy
                # if accuracy > best_accuracy:
                #     # update 'best_val_accuracy' variable to highest accuracy encountered so far-
                #     best_val_accuracy = accuracy
                #     # reset 'patience_count' variable-
                #     patience_count = 0
                # # there is no improvement in monitored metric 'accuracy'    
                # else:  
                #     # number of epochs without any improvement
                #     patience_count += 1
                    
                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    bestmodel = copy.deepcopy(model)
                    #checkdir(f"{os.getcwd()}/savesfinal/{args.arch_type}/{args.dataset}/{model_type}")
                    #torch.save(model,f"{os.getcwd()}/savesfinal/{args.arch_type}/{args.dataset}/{model_type}/{ite}_model_{args.prune_type}.pth.tar")

            # Training
            model, loss = train(model, train_loader, optimizer, criterion, epoch = iter_, warmup_scheduler=  warmup_scheduler )
            
            
            '''
            if args.warmup != 0:
                # assign warmup lr value for first args.warmup epochs
                if iter_  in range(args.warmup):
                    optimizer = lr_warmup(optimizer, lr=1/iter_per_epoch)
                # assign regular lr value for (args.warmup epoch + 1) and after
                elif iter_ == args.warmup:
                    optimizer = lr_warmup(optimizer, lr=args.lr)
            '''
            if iter_ > 0:
                scheduler.step()
                # scheduler.step(test_loss)
                # scheduler.step(accuracy)    
                # print(f"Iter value is {iter_}")
                # print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
                    
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy
            
            
            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Train Loss: {loss:.6f} Val Loss: {test_loss:.6f} Validation Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

    return bestmodel, all_loss, all_accuracy, accuracy, comp1, best_accuracy, best_val_loss


# Function for Training
def train(model, train_loader, optimizer, criterion,epoch, warmup_scheduler):
    EPS = 1e-6
    model.train()
    # total_train_loss=0.0
    total_train_loss = torch.tensor(0.0, device=device)
    # state = torch.get_rng_state()
    # cuda_state = torch.cuda.get_rng_state()

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()
        # total_train_loss += train_loss.item()
        total_train_loss += train_loss.detach()
        

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                # print(name)
                tensor = p.data
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        
        if epoch<=0:
            warmup_scheduler.step()
            # print('{:05.6f}'.format(optimizer.param_groups[0]['lr']))
        
    # torch.set_rng_state(state) 
    # torch.cuda.set_rng_state(cuda_state)
    
    # return total_train_loss/len(train_loader)
    return model, total_train_loss/torch.tensor(len(train_loader), device=device)


# Function for Testing
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    # total_test_loss = 0.0
    total_test_loss = torch.tensor(0.0, device=device)
    # correct = 0
    correct = torch.tensor(0.0, device=device)
    cuda_state = torch.cuda.get_rng_state()
    state = torch.get_rng_state()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target)
            # total_test_loss += test_loss.item()
            total_test_loss += test_loss.detach()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).sum().item()
            #correct += pred.eq(target.data.view_as(pred)).sum().detach()
            _, predicted = output.max(1)
            #total += labels_batch.size(0)
            correct += predicted.eq(target).sum().item()
            
        total_test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
    
    torch.set_rng_state(state)
    torch.cuda.set_rng_state(cuda_state)
    
    return accuracy, total_test_loss



# def training(model, args, train_loader, test_loader, model_type, bias):
    
#     print('----------------------')
#     print('model '+model_type)
    
#     print('----------------------')
    
    
#     best_accuracy = 0
#     patience_count = 0
#     previous_val_loss = float('inf')
#     accuracy=0.0
#     bestmodel = model
#     all_loss = np.zeros(args.end_iter,float)
#     all_accuracy = np.zeros(args.end_iter,float)

#     # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
#     # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    
    
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)
#     # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,65,80], gamma=0.1)
#     # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    
    
#     criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
#     # val_criterion = nn.CrossEntropyLoss(reduction='sum')

#     # Layer Looper
#     for name, param in model.named_parameters():
#         print(name, param.size())
#     # Print the table of Nonzeros in each layer
#     comp1 = print_nonzeros(model, bias=bias)
#     #comp[ite] = comp1 retun comp1de
#     pbar = tqdm(range(args.end_iter))
#     patience_count = 0
#     for iter_ in pbar:
            
#             if patience_count >= args.patience:
#                 print("\n'EarlyStopping' called!\n")
#                 break
            
#             # Frequency for Testing
#             if iter_ % args.valid_freq == 0:
                
#                 if iter_ == 0:
#                     accuracy, test_loss = test(model, test_loader, criterion)
#                 else:
#                     previous_val_loss = test_loss
#                     accuracy, test_loss = test(model, test_loader, criterion)
                    
#                 #  For checking consecutive val_loss. If patience=3, after 3 consecutive loss not improving training terminates
#                 if test_loss < previous_val_loss:
#                     # reset 'patience_count' variable-
#                     patience_count = 0
#                 # there is no improvement in monitored metric 'val_loss'    
#                 else:  
#                     # number of epochs without any improvement
#                     patience_count += 1
#                     scheduler.step()
                    
#                 # #  For checking val_loss. If patience=3, after 3 times (not consecutive) loss not improving training terminates
#                 # if test_loss > previous_val_loss:
#                 #     # number of epochs without any improvement
#                 #     patience_count += 1
#                 #     scheduler.step()
                    
#                 # Save Weights
#                 if accuracy > best_accuracy:
#                     best_accuracy = accuracy
#                     bestmodel = copy.deepcopy(model)
#                     #checkdir(f"{os.getcwd()}/savesfinal/{args.arch_type}/{args.dataset}/{model_type}")
#                     #torch.save(model,f"{os.getcwd()}/savesfinal/{args.arch_type}/{args.dataset}/{model_type}/{ite}_model_{args.prune_type}.pth.tar")

#             # Training
#             model, loss = train(model, train_loader, optimizer, criterion)
#             all_loss[iter_] = loss
#             all_accuracy[iter_] = accuracy
            
            
#             # Frequency for Printing Accuracy and Loss
#             if iter_ % args.print_freq == 0:
#                 pbar.set_description(
#                     f'Train Epoch: {iter_}/{args.end_iter} Train Loss: {loss:.6f} Val Loss: {test_loss:.6f} Validation Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')       

#     return bestmodel, all_loss, all_accuracy, accuracy, comp1, best_accuracy, previous_val_loss