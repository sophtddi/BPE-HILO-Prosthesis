"""
This file contains utility functions for handling user preferences, updating models, and managing data, used both in ID.py and exploration.py:

- compare_objective_values(pair):
Prompts the user to compare two configurations and provide a preference.

- user_preference(pair):
Determines the user's preferred configuration from a pair and updates the current best configuration.

- no_preference(method, previous_pair, unliked_previous, dim=dim):
In case the user has no preference, define the way to store data for the model depending on 'method'

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
import os
import numpy as np
from scipy.stats import qmc

from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood

from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.fit import fit_gpytorch_mll

from config import lower_bounds, upper_bounds, noise_likelihood, dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
np.set_printoptions(precision=2, suppress=True)
    
def compare_objective_values(pair):
    """
    For a given pair = [x1, x2], prompt to ask user preference
    0 -> no preference -> return [-1, -1], call no_preference function defined in exploration_function
    1 -> x1
    2 -> x2

    Args:
        pair: torch.tensor[[x1, x2]] containing the 2 proposed configurations in their normalized form
    
    Returns:
        compt: torch.tensor: [[0, 1]] if first configuration is prefered,
                             [[1, 0]] if second configuration is prefered,
    """
   
    assert pair.shape[0] == 2

    print(f"Send option 1 to user: {translate_param(pair[0]).tolist()}")
    # # time.sleep(3)
    print(f"Send option 2 to user: {translate_param(pair[1]).tolist()}")

    ## Ask user preference
    while True:
        choice = input("Preference? (0, 1 or 2 ): ")
        if choice in ['0', '1', '2']:
            break
        print("Enter 1 or 2.")
    
    if choice == '0': # if no preference, data update manage by no_preference function
        choice = 0
        return torch.tensor([[-1, -1]]).long()  
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
    loser_index = 1 if comp[0][0] == 0 else 0    # if comp = [0,1] then first configuration is preferred
    unliked_config = pair[loser_index:loser_index+1]  # extract the unliked configuration
    return comp, current_best, unliked_config           

def no_preference(method, previous_pair, unliked_previous, dim=dim):
    """ 
    In case the user has no preference, define the way to store data for the model depending on method
    """
    if method == 'random':              # (a > c), (b > c) with c a random point
        challenger = torch.rand(dim)
        next_pair = torch.stack([previous_pair[0], challenger])
        other_next_pair = torch.stack([previous_pair[1], challenger])

    elif method == 'previous':          # (a > c), (b > c) with c the unliked point of previous iteration
        challenger = unliked_previous.squeeze(0)
        next_pair = torch.stack([previous_pair[0], challenger])
        other_next_pair = torch.stack([previous_pair[1], challenger])
    
    elif method == 'double':            # (a > b) and (b > a)
        next_pair = torch.stack([previous_pair[0], previous_pair[1]])
        other_next_pair = torch.stack([previous_pair[1], previous_pair[0]])

    return next_pair, other_next_pair

def update_model(model, mll, likelihood, train_x, train_comp, noise_likelihood):
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
        # Model parametrization                                            # lengthscale initialized at previous value
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=4)
        kernel = ScaleKernel(base_kernel)                                           # variance initialized at previous value
        likelihood = PairwiseLogitLikelihood(noise_likelihood=noise_likelihood)
        new_model = PairwiseGP(
            train_x,
            train_comp,
            covar_module=kernel,
            likelihood=likelihood,
            jitter=1e-5
        )   

        # Model training
        new_mll = PairwiseLaplaceMarginalLogLikelihood(new_model.likelihood, new_model) # fit the new model
        fit_gpytorch_mll(new_mll)
        
        return new_model, new_mll, likelihood

    except Exception as e:      # if the model fitting fail, keep the same model
        print(f"WARNING: ERROR IN TRAINING : {e}")
        return model, mll, likelihood
    
def append_data(x_next, x_train, comp_next, comp_train, tol=1e-4):
    """
    Adds a new observation (a pairwise comparison) to the training data.
    The new observation has to be translated from its original format ([0,1] or [1,0]) to indices format ([i,j] or [j,i])
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
    # print(f"pair {x_next} comparison is {new_comp_next}")
    return new_x_train, new_comp_train

# def translate_param(x, lower_bounds=lower_bounds, upper_bounds=upper_bounds):
#     # Ensure x is 2D
#     x = np.atleast_2d(x)
#     x = np.clip(x, 0.0, 1.0)

#     return qmc.scale(x, lower_bounds, upper_bounds).flatten()

def translate_param(x, lower_bounds=lower_bounds, upper_bounds=upper_bounds):
    # Ensure x is 2D
    x = np.atleast_2d(x)
    x = np.clip(x, 0.0, 1.0)

    # Scale to real domain
    scaled = qmc.scale(x, lower_bounds, upper_bounds)

    # Allocate output array
    rounded = np.empty_like(scaled)

    # Apply rounding rules:
    # dim 0: 1 decimal
    rounded[:, 0] = np.round(scaled[:, 0], 1)

    # dim 1: 2 decimals
    rounded[:, 1] = np.round(scaled[:, 1], 2)

    # dim 2: 1 decimal
    rounded[:, 2] = np.round(scaled[:, 2], 1)

    # dim 3: 2 decimals
    rounded[:, 3] = np.round(scaled[:, 3], 2)

    # Return flattened only for a single point
    return rounded.flatten() if x.shape[0] == 1 else rounded


def inverse_translate_param(x, lower_bounds=lower_bounds, upper_bounds=upper_bounds):
    """
    Convert a point or array from the real domain defined by `lower_bounds`/`upper_bounds`
    back to the normalized [0,1]^d space.

    Accepts `torch.Tensor` or array-like inputs. Returns a numpy array (flattened for a
    single point, else 2D array).
    """
    # Accept torch tensors
    if x is None:
        raise ValueError("inverse_translate_param: input x is None")
    if hasattr(x, "detach"):
        try:
            x = x.detach().cpu().numpy()
        except Exception:
            x = np.asarray(x)

    x = np.asarray(x, dtype=float)
    x = np.atleast_2d(x)

    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)
    if lb.shape[0] != x.shape[1] or ub.shape[0] != x.shape[1]:
        raise ValueError("inverse_translate_param: dimension mismatch between bounds and input")

    denom = ub - lb
    if np.any(denom == 0):
        raise ValueError("inverse_translate_param: zero width in bounds, cannot normalize")

    normalized = (x - lb) / denom
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.flatten() if normalized.shape[0] == 1 else normalized


def save_model(best_config, option1, option2, pref, train_x, train_comp, visited_pairs, model, likelihood, mll, save_dir, test_id, i='_'):
    if model == None:
        torch.save({
        'best_point': best_config,
        'option1': option1,
        'option2': option2,
        'preference': pref,
        'train_x': train_x,
        'train_comp': train_comp,
        'visited_pairs': visited_pairs,
        }, os.path.join(save_dir, f'{test_id}{i}data.pt'))
    else: 
        torch.save({
        'best_point': best_config,
        'option1': option1,
        'option2': option2,
        'preference': pref,
        'train_x': train_x,
        'train_comp': train_comp,
        'visited_pairs': visited_pairs,
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        "model": model,
        "likelihood": likelihood,
        "mll": mll,
        }, os.path.join(save_dir, f'{test_id}{i}data.pt'))

def retrain_model(file_path, noise_likelihood=noise_likelihood):
    checkpoint = torch.load(file_path, weights_only=False)
    
    # reconstruire kernel & likelihood
    # base_kernel = MaternKernel(nu=2.5)
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=4)

    kernel = ScaleKernel(base_kernel)
    likelihood = PairwiseLogitLikelihood(noise_likelihood=noise_likelihood)
    
    # reconstruire le modèle avec les données sauvegardées
    restored_model = PairwiseGP(
        datapoints=checkpoint["train_x"],
        comparisons=checkpoint["train_comp"],
        covar_module=kernel,
        likelihood=likelihood,
        jitter=1e-5,
    )
    
    # recharger les poids
    # restored_model.load_state_dict(checkpoint["model_state_dict"])
    # restored_model.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

    new_mll = PairwiseLaplaceMarginalLogLikelihood(restored_model.likelihood, restored_model) # fit the new model
    fit_gpytorch_mll(new_mll)

    return restored_model, likelihood, checkpoint

def reload_model(file_path):
    checkpoint = torch.load(file_path, weights_only=False)    
    model = checkpoint["model"]
    likelihood = checkpoint["likelihood"]
    return model, likelihood, checkpoint
