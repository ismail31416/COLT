import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.prune as prune
import random
import os


# def original_initialization(model, mask_temp, initial_state_dict):
    
#     global step
#     step = 0
#     for name, param in model.named_parameters(): 
#         if "weight" in name: 
#             weight_dev = param.device
#             param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
#             step = step + 1
#         if "bias" in name:
#             param.data = initial_state_dict[name]
#     step = 0
#     return model

def weight_rewinding(pruned_model, unpruned_model, bias=True):
    """
    Copy weights of from an unpruned network to a pruned network
    
    Everything including bias is rewinded.
    """
    
    with torch.no_grad():
        for pruned_module, unpruned_module in zip(pruned_model.modules(), unpruned_model.modules()):
            if isinstance(pruned_module, nn.Linear) or  isinstance(pruned_module, nn.Conv2d) or isinstance(pruned_module, nn.BatchNorm2d):
                from prune import check_pruned
                if check_pruned(pruned_module, bias=bias):
                    # print(f"Rewinding!!!! {pruned_module}")
                    pruned_module.weight_orig.copy_(unpruned_module.weight)
                    # prune.remove(pruned_module, 'weight') 
                    if bias:
                        pruned_module.bias_orig.copy_(unpruned_module.bias)
                        # prune.remove(pruned_module, 'bias')
                    else:
                        if pruned_module.bias is not None:
                            pruned_module.bias.copy_(unpruned_module.bias)
                else:
                    pruned_module.weight.copy_(unpruned_module.weight)
                    if pruned_module.bias is not None:
                        pruned_module.bias.copy_(unpruned_module.bias)
    
    return pruned_model  
    

# Function to make an empty mask of the same size as the model
# def make_mask(model):
    
#     global step
#     step = 0
#     for name, param in model.named_parameters(): 
#         if 'weight' in name:
#             step = step + 1
#     mask = [None]* step 
#     step = 0
#     for name, param in model.named_parameters(): 
#         if 'weight' in name:
#             tensor = param.data.cpu().numpy()
#             mask[step] = np.ones_like(tensor)
#             step = step + 1
#     step = 0
    
#     return mask 

#Print table of zeros and non-zeros count
# def print_nonzeros(model):
#     nonzero = total = 0.0
#     for name, p in model.named_parameters():
#         tensor = p.data.cpu().numpy()
#         nz_count = np.count_nonzero(tensor)
#         total_params = np.prod(tensor.shape)
#         nonzero += nz_count
#         total += total_params
#         print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    
#     prune_rate = 100.0 * (total-nonzero) / total
#     print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({prune_rate:6.2f}% pruned)')
    
#     return (round((nonzero/total)*100.0,1))

def print_nonzeros(model, bias=True):
    nonzero = total = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            from prune import check_pruned
            if check_pruned(module, bias=bias):
                tensor = module.weight_mask.detach().cpu().numpy()
            else:
                tensor = module.weight.detach().cpu().numpy()
                
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    
    prune_rate = 100.0 * (total-nonzero) / total
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({prune_rate:6.2f}% pruned)')
    
    return (round((nonzero/total)*100.0,1))


def count_matching_weights(ticket1, ticket2, layer):
    '''
    function to count only matching weghts between two layer of two different weights

    Inputs:
    'ticket1' is the numpy array of weight of model1
    'ticket2' is the numpy array of weight of model2
    'layer' is the the layer number of both the models

    Returns:
    integer value, which is the total number of matching weights 
    '''
  

    # For dense layer 1, we only take the weigths that match in both winning tickets.
    # So we created a mask, where 1 denotes matching weights and 0 denotes non-matching weights
    mask = np.equal(ticket1[layer], ticket2[layer])
    mask = mask.astype(int)
    #print(mask)

    matching_weights = mask*ticket2[layer]
    
    weight1 = np.count_nonzero(matching_weights)
    weight2 = np.count_nonzero(ticket1[layer])

    return (weight1/weight2 if weight2 else 0)*100


def only_matching_weights(ticket1, ticket2, name):
  
    # For dense layer 1, we only take the weigths that match in both winning tickets.
    # So we created a mask, where 1 denotes matching weights and 0 denotes non-matching weights

    mask = np.equal(ticket1, ticket2)
    mask = mask.astype(int)
    #mask = np.equal(ticket1, ticket2)
    matching_weights = mask*ticket1
    
    weight1 = np.count_nonzero(matching_weights)
    weight2 = np.count_nonzero(ticket1)
    percent = (weight1/weight2 if weight2 else 0)*100
    print("Matching weights ---",name,"---   ",percent,"%")
    
    return matching_weights

# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
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