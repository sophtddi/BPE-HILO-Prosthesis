"""
This file contains specific functions used during the exploration phase of the experiment in the exploration.py file

- problem_exploration(xs, pairs_to_test, l, length_bounds=length_bounds, out_bounds=out_bounds):
Ask user preference over a some pairs to build an inital model.
"""
import torch
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import Interval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

from core_functions import user_preference, append_data
from config import length_bounds, out_bounds                # defines the bounds for lengthscale and outputscale of the model


def problem_exploration(xs, pairs_to_test, l, length_bounds=length_bounds, out_bounds=out_bounds):
    """
    Model initialisation: exploration of preference between predefined pairs of configuration
    The model is a GP using a RBF kernel with lengthscale l build over user preference to minimize PairwiseLaplaceMarginalLogLikelihood

    Args:
        xs: grid of all the possible configuration
        n_steps: number of comparison to initialize the model
        l: lengthscale of the model

    Returns:
        train_x: list of visited points
        train_comp: list of comparison output ([0,1] or [1,0])
        model: model
        mll: model likelihood
        visited_pairs: list of visited pairs
        l: lengthscale of the model
    """
    visited_pairs = [] 

    # Translate the configuration in their real values
    indices_pairs = []
    for x1, x2 in pairs_to_test:
        x1_tensor = torch.tensor(x1)
        x2_tensor = torch.tensor(x2)
        idx1 = torch.isclose(xs, x1_tensor, rtol=1e-4, atol=1e-4).all(dim=1).nonzero(as_tuple=True)
        idx2 = torch.isclose(xs, x2_tensor, rtol=1e-4, atol=1e-4).all(dim=1).nonzero(as_tuple=True)
        indices_pairs.append((idx1[0].item(), idx2[0].item()))

    # # For each pair, ask user preference
    idx1, idx2 = indices_pairs[0]
    train_x = torch.stack([xs[idx1], xs[idx2]], dim=0)  # pair to compare
    visited_pairs.append(train_x)
    print(f"\nEXPLORATION 1: {(train_x[0])} vs {(train_x[1])} ")
    train_comp, _ = user_preference(train_x)            # ask user preference

    for i in range(1, len(indices_pairs)):
        idx1, idx2 = indices_pairs[i]                       
        next_x = torch.stack([xs[idx1], xs[idx2]], dim=0)   # pair to compare
        visited_pairs.append(next_x)
        print(f'\nEXPLORATION {i+1}: {(next_x[0])} vs {(next_x[1])}')
        next_comp, _ = user_preference(next_x)              # ask user preference
        train_x, train_comp = append_data(next_x, train_x, next_comp, train_comp) # store the visited pair to build the model
    print(f'the model is trained over train_x {train_x} and train_comp {train_comp}')
    
    # Model initalisation
    base_kernel = RBFKernel(lengthscale_constraint=Interval(*length_bounds))
    base_kernel.lengthscale = l 
    kernel = ScaleKernel(base_kernel, outputscale_constraint=Interval(*out_bounds))

    model = PairwiseGP(
        train_x, 
        train_comp, 
        covar_module=kernel,
        jitter=1e-2
    )
    # Model fitting
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    new_l = model.covar_module.base_kernel.lengthscale
    new_var = model.covar_module.outputscale
    return train_x, train_comp, model, mll, visited_pairs, new_l, new_var

