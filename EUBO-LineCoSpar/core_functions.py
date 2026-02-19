"""
This file contains utility functions for handling user preferences, updating models, and managing data, used both in exploration.py and optimization.py:

- compare_objective_values(pair):
Prompts the user to compare two configurations and provide a preference.

- user_preference(pair):
Determines the user's preferred configuration from a pair and updates the current best configuration.

- update_model(model, mll, train_x, train_comp):
Updates the model using new observed preferences.

- append_data(x_next, x_train, comp_next, comp_train):
Adds new observations to the training data.

- translate_param(x):
Translates normalized values to the range of defined parameters.

- save_model(current_best, challenger, train_x, train_comp, model, mll, lengthscale, var, visited_pairs, save_dir, test_id, i):
Saves the current state of the model and related data.
"""

import torch
import random 
import os
import time

from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import Interval
from botorch.fit import fit_gpytorch_mll

from config import length_bounds, out_bounds, prior_scale
from config import params1, params2, params3, params4, n_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


def compare_objective_values(pair):
    """
    For a given pair = [x1, x2], prompt to ask user preference
    0 -> no preference -> random
    1 -> x1
    2 -> x2

    Args:
        pair: torch.tensor[[x1, x2]] containing the 2 proposed configurations in their normalized form
    
    Returns:
        compt: torch.tensor: [[0, 1]] if first configuration is prefered,
                             [[1, 0]] if second configuration is prefered,
    """
   
    assert pair.shape[0] == 2

    print(f"Send option 1 to user: {translate_param(pair[0])}")    # Print the configuration
    time.sleep(3)
    print(f"Send option 2 to user: {translate_param(pair[1])}")    # Print the configuration

    ## Ask user preference
    while True:
        choice = input("Preference? (0, 1 or 2 ): ")
        if choice in ['0', '1', '2']:
            break
        print("Enter 1 or 2.")
    
    if choice == '0': # if no preference, randomly select a preference
        random_choice = random.choice(['1', '2'])
        print(f"Random choice: option {random_choice} is prefered")
        choice = random_choice
    if choice == '1':
        return torch.tensor([[0, 1]]).long()  # user pref option 1
    else: # no pref or pref 2
        return torch.tensor([[1, 0]]).long()  # user pref option 2 

def user_preference(pair):
    """
    Give user preference within a pair, and actualize the prefered configuration

    Args:
        pair: torch.tensor[[x1], [x2]] containing the 2 configurations to compare

    Returns:
        answer: torch.tensor: [[0, 1]] if x1 is preferred,
                              [[1, 0]] if x2 is preferred,
        current_best: preferred configuration so far  
    """
    comp = compare_objective_values(pair)           # prompt user preference and store it 
    winner_index = 0 if comp[0][0] == 0 else 1      # if comp = [0,1] then first configuration is preferred
    current_best = pair[winner_index:winner_index+1]  # extract the prefered configuration

    return comp, current_best           

def update_model(model, mll, train_x, train_comp, length_bounds=length_bounds, out_bounds=out_bounds, prior_scale=prior_scale):
    """
    Update the model by using the new observed preference

    Args:
        model: model
        mll: model likelihood
        train_x: visited configurations
        train_comp: observed preferences 

    Returns:
        new_model: updated model
        new_mll: updated model likelihood
    """
    
    try:
        # Model parametrization
        prev_l = model.covar_module.base_kernel.lengthscale.detach().clone()            # recover model lenghtscale
        prev_out = model.covar_module.outputscale.detach().clone()                      # recover model variance
        lengthscale_prior = LogNormalPrior(loc=torch.log(prev_l), scale=prior_scale)    # lengthscale can be updated given a LogNormal distribution around previous lengthscale
        base_kernel = RBFKernel(lengthscale_prior=lengthscale_prior, lengthscale_constraint=Interval(*length_bounds))    # lengthscale is contrained within some defined length_bounds
        base_kernel.lengthscale = prev_l/2                                              # lengthscale initialized at previous value
      
        outputscale_prior = LogNormalPrior(loc=torch.log(prev_out), scale=prior_scale)  # variance can be updated given a LogNormal distribution around previous variance
        kernel = ScaleKernel(base_kernel, outputscale_prior=outputscale_prior,outputscale_constraint=Interval(*out_bounds)) # outputscale is contrained within some defined out_bounds
        kernel.outputscale = prev_out                                                   # variance initialized at previous value
        
        new_model = PairwiseGP(
            train_x,
            train_comp,
            covar_module=kernel,
            jitter=1e-5,
        )   

        # Model training
        new_mll = PairwiseLaplaceMarginalLogLikelihood(new_model.likelihood, new_model) # fit the new model
        fit_gpytorch_mll(new_mll)
        
        return new_model, new_mll

    except Exception as e:      # if the model fitting fail, keep the same model
        print(f"WARNING: ERROR IN TRAINING : {e}")
        return model, mll
    
def append_data(x_next, x_train, comp_next, comp_train, tol=1e-4):
    """
    Adds a new observation (a pairwise comparison) to the training data.
    The new observation has to be translated from its original format ([0,1] or [1,0]) to indices format ([i,j] or [j,ji])
    with i and j, the respective indices of the 2 configurations in new_x_train

    Args:
        x_next: Tensor of shape (2, d). Contains the two configurations being compared: 
                - x_next[0]: current_best
                - x_next[1]: challenger
        x_train: Tensor of previously visited configurations
        comp_next: Tensor of shape (1, 2). The preference result between current_best and challenger
                   Format: [0, 1] if the second config is preferred, [1, 0] otherwise
        comp_train: Tensor of previously observed preferences (list of [i, j] index pairs)

    Returns:
        new_x_train: Updated list of visited configurations (x_train with any new configs from x_next added)
        new_comp_train: Updated list of preferences (comp_train with comp_next added but translated into index pairs [i,j])
    """

    n = x_train.shape[-2]                     # number of aleady visited configuration

    new_x_train = x_train.clone()             # new_x_train is initialized with the already observed points stored in x_train
    new_comp_next = comp_next.clone() + n     # elements of x_next are added at the end of x_train so their indices in new_x_train are n and n+1 (it's just an initalisation)

    # Now we will check for the duplicates

    # Check if current_best (x_next[0]) is already in x_train
    n_dups = 0  # Counter to track if current_best is already in x_train
    dup_ind = torch.where(torch.all(torch.isclose(x_train, x_next[0], atol=tol), dim=1))[0] # look for the index of current_best in new_x_train 
    if dup_ind.nelement() == 0:
        new_x_train = torch.cat([x_train, x_next[0].unsqueeze(-2)])                         # if not found, append it to x_train in new_x_train
    else:
        new_comp_next = torch.where(new_comp_next == n, dup_ind, new_comp_next - 1)         # if found, update comp_next to refer to existing index in new_x_train
        n_dups += 1

    # Check if challenger (x_next[1]) is already in x_train or was just added
    dup_ind = torch.where(torch.all(torch.isclose(new_x_train, x_next[1], atol=tol), dim=1))[0] # look for the index of challenger in new_x_train 
    if dup_ind.nelement() == 0:
        new_x_train = torch.cat([new_x_train, x_next[1].unsqueeze(-2)]) # if not found, append to new_x_train
    else:
        new_comp_next = torch.where(
            new_comp_next == n + 1 - n_dups, dup_ind, new_comp_next)    # if found, update comp_next to refer to existing index in new_x_train
        
    new_comp_train = torch.cat([comp_train, new_comp_next])     # append the new comparison result to the already observed comparisons

    return new_x_train, new_comp_train

def translate_param(x, params1=params1, params2=params2, params3=params3, params4=params4, n_grid=n_grid):
    """
    Translate the normalized 4 values tensor in a tensor within the range of values of the defined parameters
    Args:
        x: tensor with normalized values (between 0 and 1)
        
    Returns:
        scaled_config: tensor scaled back to the parameters range of values
    """
    if x.ndim == 2 and x.shape[0] == 1:
        x = x.squeeze(0)  # remove batch dimension (1, 4) → (4,)

    idxs =  (x * (n_grid-1)).long()                         # transform normalized values into indices 
    param1     = int(params1[idxs[0]].item())               # use the translated index to find the value of param1        
    param2     = round(float(params2[idxs[1]]), 1)          # use the translated index to find the value of param2   
    param3 = int(params3[idxs[2]].item())                   # use the translated index to find the value of param3    
    param4     = round(float(params4[idxs[3]]), 1)          # use the translated index to find the value of param4

    scaled_config = [param1, param2, -param3, param4]       # tensor scaled back to the parameters range of values

    return scaled_config

def save_model(current_best, challenger, train_x, train_comp, model, mll, lengthscale, var, visited_pairs,save_dir, test_id, i='_'):
    torch.save({
    'best_point': current_best,
    'challenger': challenger,
    'train_x': train_x,
    'train_comp': train_comp,
    'model': model,
    'mll': mll,
    'lengthscale':lengthscale,
    'var': var,
    'model_state_dict': model.state_dict(),
    'visited_pairs': visited_pairs}, 
    os.path.join(save_dir, f'{test_id}{i}data.pt'))