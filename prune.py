import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from weights import only_matching_weights
import torch
import random
import os


# def colt(model1,model2):    
    
#     for (name1, param1), (name2,param2) in zip(model1.named_parameters(),model2.named_parameters()): 
#         if "weight" in name1: 
#             device = param1.device
#             ticket1 = param1.data.cpu().detach().numpy()
#             ticket2 = param2.data.cpu().detach().numpy()
#             overlap = only_matching_weights(ticket1, ticket2, name1)
#             param1.data = torch.from_numpy(overlap).float().to(device)
#             param2.data = torch.from_numpy(overlap).float().to(device)      
    
#     return model1,model2

def check_pruned(module, bias=True):
    """Check if a module was pruned.
    We require both the bias and the weight to be pruned.
    Returns
    -------
    bool
        True if the model has been pruned.
    """
    params = {param_name for param_name, _ in module.named_parameters()}
    
    if bias == True:
        expected_params = {"weight_orig", "bias_orig"}
        return params == expected_params
    else:
        expected_params1 = {"weight_orig"}
        expected_params2 = {"weight_orig", "bias"}
        return params == expected_params1 or params == expected_params2


def load_weights(model1,model2, bias = False):


    with torch.no_grad():
        for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()): 
            if isinstance(module1, nn.Linear) or  isinstance(module1, nn.Conv2d) or isinstance(module1, nn.BatchNorm2d):
                device = module1.weight.device

                if check_pruned(module1, bias=bias):
                    ticket1_weight = module1.weight_mask.detach().cpu().numpy()
                    ticket2_weight = module2.weight_mask.detach().cpu().numpy()

                    overlap_weight = only_matching_weights(ticket1_weight, ticket2_weight, name1)

                    module1.weight_mask.copy_(torch.from_numpy(overlap_weight).float().to(device))
                    module2.weight_mask.copy_(torch.from_numpy(overlap_weight).float().to(device))

                    if bias:
                        ticket1_bias = module1.bias_mask.detach().cpu().numpy()
                        ticket2_bias = module2.bias_mask.detach().cpu().numpy()

                        overlap_bias = only_matching_weights(ticket1_bias, ticket2_bias, name1)

                        module1.bias_mask.copy_(torch.from_numpy(overlap_bias).float().to(device))
                        module2.bias_mask.copy_(torch.from_numpy(overlap_bias).float().to(device))


                else:
                    ticket1_weight = module1.weight.detach().cpu().numpy()
                    ticket2_weight = module2.weight.detach().cpu().numpy()

                    overlap_weight = only_matching_weights(ticket1_weight, ticket2_weight, name1)

                    module1.weight.copy_(torch.from_numpy(overlap_weight).float().to(device))
                    module2.weight.copy_(torch.from_numpy(overlap_weight).float().to(device))

                    if bias:
                        ticket1_bias = module1.bias.detach().cpu().numpy()
                        ticket2_bias = module2.bias.detach().cpu().numpy()

                        overlap_bias = only_matching_weights(ticket1_bias, ticket2_bias, name1)

                        module1.bias.copy_(torch.from_numpy(overlap_bias).float().to(device))
                        module2.bias.copy_(torch.from_numpy(overlap_bias).float().to(device)) 
                
    
    return model1,model2

def colt(model1,model2,bias=False):  
    
    """
    Pruned overlapping weights between two models
    
    If bias=True, 
    
    """
    with torch.no_grad():
        for (name1, module1), (name2, module2) in zip(model1.named_modules(), model2.named_modules()): 
            if isinstance(module1, nn.Linear) or  isinstance(module1, nn.Conv2d) or isinstance(module1, nn.BatchNorm2d):
                device = module1.weight.device

                if check_pruned(module1, bias=bias):
                    ticket1_weight = module1.weight_mask.detach().cpu().numpy()
                    ticket2_weight = module2.weight_mask.detach().cpu().numpy()

                    overlap_weight = only_matching_weights(ticket1_weight, ticket2_weight, name1)

                    module1.weight_mask.copy_(torch.from_numpy(overlap_weight).float().to(device))
                    module2.weight_mask.copy_(torch.from_numpy(overlap_weight).float().to(device))

                    if bias:
                        ticket1_bias = module1.bias_mask.detach().cpu().numpy()
                        ticket2_bias = module2.bias_mask.detach().cpu().numpy()

                        overlap_bias = only_matching_weights(ticket1_bias, ticket2_bias, name1)

                        module1.bias_mask.copy_(torch.from_numpy(overlap_bias).float().to(device))
                        module2.bias_mask.copy_(torch.from_numpy(overlap_bias).float().to(device))


                else:
                    ticket1_weight = module1.weight.detach().cpu().numpy()
                    ticket2_weight = module2.weight.detach().cpu().numpy()

                    overlap_weight = only_matching_weights(ticket1_weight, ticket2_weight, name1)

                    module1.weight.copy_(torch.from_numpy(overlap_weight).float().to(device))
                    module2.weight.copy_(torch.from_numpy(overlap_weight).float().to(device))

                    if bias:
                        ticket1_bias = module1.bias.detach().cpu().numpy()
                        ticket2_bias = module2.bias.detach().cpu().numpy()

                        overlap_bias = only_matching_weights(ticket1_bias, ticket2_bias, name1)

                        module1.bias.copy_(torch.from_numpy(overlap_bias).float().to(device))
                        module2.bias.copy_(torch.from_numpy(overlap_bias).float().to(device)) 
                
    
    return model1,model2

def colt2(models, partition):
    

    for i in range(1,partition):
       
        _,models[i] = colt(models[i-1],models[i],bias = False)

    for i in range(partition-1):
        models[i] = load_weights(models[i],models[-1])

    return models

# # Prune by Percentile module
# def prune_by_percentile(model,mask,percent, **kwargs):
        
#         global step
        
        
#         # Calculate percentile value
#         step = 0
#         for name, param in model.named_parameters():

#             # We do not prune bias term
#             if 'weight' in name:
#                 tensor = param.data.cpu().numpy()
#                 alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
#                 percentile_value = np.percentile(abs(alive), percent)

#                 # Convert Tensors to numpy and calculate
#                 weight_dev = param.device
#                 new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
#                 # Apply new weight and mask
#                 param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
#                 mask[step] = new_mask
#                 step += 1
#         step = 0
        
#         return mask 


    
# def prune_rate_calculator(model):
#     nonzero = total = 0.0
#     for name, p in model.named_parameters():
#         tensor = p.data.cpu().numpy()
#         nz_count = np.count_nonzero(tensor)
#         total_params = np.prod(tensor.shape)
#         nonzero += nz_count
#         total += total_params
    
#     rate = 100.0 * (total-nonzero) / total
        
#     return rate

def prune_rate_calculator(model, bias=True):
    nonzero = total = 0.0
    for module in model.modules():
        if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            if check_pruned(module, bias=bias):
                # print(f"YES! {module} pruned!!")
                tensor = module.weight_mask.detach().cpu().numpy()
            else:
                tensor = module.weight.detach().cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params            

    rate = 100. * (total-nonzero) / total
    return rate

    
def lth_local_unstructured_pruning(model, output_class, bias=True, **layers):
    """
    
    Does layerwise unstructured pruning for a given model
    
    If bias=True, also prunes bias
    
    """
    
    for name, module in model.named_modules():
        
        if isinstance(module, nn.Linear):
            # if module.weight.shape[1] == output_class:
            if module.out_features == output_class:
                prune.l1_unstructured(module, name = 'weight', amount = layers['output'])
                # prune.remove(module, 'weight')
                if bias:
                    prune.l1_unstructured(module, name = 'bias', amount = layers['output'])
                     # prune.remove(module, 'bias')  
            else:
                prune.l1_unstructured(module, name = 'weight', amount = layers['linear'])
                # prune.remove(module, 'weight')
                if bias:
                    prune.l1_unstructured(module, name = 'bias', amount = layers['linear'])
                    # prune.remove(module, 'bias')                   
                      
        elif isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name = 'weight', amount = layers['conv'])
            # prune.remove(module, 'weight')
            if bias:
                prune.l1_unstructured(module, name = 'bias', amount = layers['conv'])
                # prune.remove(module, 'bias')
        
        elif isinstance(module, nn.BatchNorm2d):
            prune.l1_unstructured(module, name = 'weight', amount = layers['batchnorm'])
            # prune.remove(module, 'weight')
            if bias:
                prune.l1_unstructured(module, name = 'bias', amount = layers['batchnorm'])
                # prune.remove(module, 'bias')
            
            
    return model



def lth_global_unstructured_pruning(model, output_class, bias=True, **layers):
    """
    
    Does global unstructured pruning for given model
    
    If bias=True, also prunes bias
    
    """
    all_linear_module_weights = []
    all_linear_module_biases = []
    all_output_module_weights = []
    all_output_module_biases = []
    all_conv_module_weights = []
    all_conv_module_biases = []
    all_batchnorm_module_weights = []
    all_batchnorm_module_biases = []
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.out_features == output_class:
                all_output_module_weights.append((module, 'weight'))
                all_output_module_biases.append((module, 'bias')) 
            else:
                all_linear_module_weights.append((module, 'weight'))
                all_linear_module_biases.append((module, 'bias'))            
        
        elif isinstance(module, nn.Conv2d):
            all_conv_module_weights.append((module, 'weight'))
            all_conv_module_biases.append((module, 'bias'))
            
        elif isinstance(module, nn.BatchNorm2d):
            all_batchnorm_module_weights.append((module, 'weight'))
            all_batchnorm_module_biases.append((module, 'bias'))            

    if 'linear' in layers.keys() and len(all_linear_module_weights)!=0 and layers['linear']!=0.0:
        print('L')
        prune.global_unstructured(parameters=all_linear_module_weights, pruning_method=prune.L1Unstructured, amount=layers['linear'])
        if bias:
            prune.global_unstructured(parameters=all_linear_module_biases, pruning_method=prune.L1Unstructured, amount=layers['linear'])
        
        # for module, _ in all_linear_module_weights:
        #     prune.remove(module, 'weight')
        # for module, _ in all_linear_module_biases:
        #     prune.remove(module, 'bias')

    if 'output' in layers.keys() and len(all_output_module_weights)!=0 and layers['output']!=0.0:
        print('O')
        prune.global_unstructured(parameters=all_output_module_weights, pruning_method=prune.L1Unstructured, amount=layers['output'])
        if bias:
            prune.global_unstructured(parameters=all_output_module_biases, pruning_method=prune.L1Unstructured, amount=layers['output'])  

        # for module, _ in all_output_module_weights:
        #     prune.remove(module, 'weight')
        # for module, _ in all_output_module_biases:
        #     prune.remove(module, 'bias')
            
    if 'conv' in layers.keys() and len(all_conv_module_weights)!=0 and layers['conv']!=0.0:
        print('C')
        prune.global_unstructured(parameters=all_conv_module_weights, pruning_method=prune.L1Unstructured, amount=layers['conv'])
        if bias:
            prune.global_unstructured(parameters=all_conv_module_biases, pruning_method=prune.L1Unstructured, amount=layers['conv'])  
        
        # for module, _ in all_conv_module_weights:
        #     prune.remove(module, 'weight')
        # for module, _ in all_conv_module_biases:
        #     prune.remove(module, 'bias')
        
    if 'batchnorm' in layers.keys() and len(all_batchnorm_module_weights)!=0 and layers['batchnorm']!=0.0:
        print('B')
        prune.global_unstructured(parameters=all_batchnorm_module_weights, pruning_method=prune.L1Unstructured, amount=layers['batchnorm'])
        if bias:
            prune.global_unstructured(parameters=all_batchnorm_module_biases, pruning_method=prune.L1Unstructured, amount=layers['batchnorm']) 
        
        # for module, _ in all_batchnorm_module_weights:
        #     prune.remove(module, 'weight')
        # for module, _ in all_batchnorm_module_biases:
        #     prune.remove(module, 'bias')
    
    return model
    
    