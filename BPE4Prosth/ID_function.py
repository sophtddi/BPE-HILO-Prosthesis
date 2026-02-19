"""
This file contains specific functions used during the initial design phase of the experiment in the ID.py file

    - initial_design(pairs_to_test, noise_likelihood): 
Ask user preference over some pairs to build an inital model
"""


import torch
import numpy as np
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.likelihoods.pairwise import PairwiseLogitLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
np.set_printoptions(precision=2, suppress=True)
torch.set_default_dtype(torch.double)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

from core_function import user_preference, append_data


def initial_design(pairs_to_test, dim, noise_likelihood):
    """
    Model initialisation: ask user preference between predefined pairs of configuration
    The model is a GP using a RBF kernel build over user preference to minimize PairwiseLaplaceMarginalLogLikelihood

    Args:
        pairs_to_test: random pair that fill the space 
        noise_likelihood: model user inconsistency in the likelihood object
    Returns:
        train_x: list of visited points
        train_comp: list of comparison output ([0,1] or [1,0])
        model: model
        mll, likelihood: model likelihood
        visited_pairs: list of visited pairs
        new_l, new_var: lengthscale and variance of the model
    """
    visited_pairs = [] 
    train_x = torch.empty((0, dim), dtype=torch.double)
    train_comp = torch.empty((0, 2), dtype=torch.long)

    for i in range(len(pairs_to_test)):
        next_x = torch.stack([pairs_to_test[i][0], pairs_to_test[i][1]], dim=0)   # pair to compare
        visited_pairs.append(next_x)
        print(f'\nEXPLORATION {i+1}: {(next_x[0])} vs {(next_x[1])}')
        next_comp, _, _ = user_preference(next_x)              # ask user preference
        train_x, train_comp = append_data(next_x, train_x, next_comp, train_comp) # store the visited pair to build the model

    # Model initalisation
    base_kernel = MaternKernel(nu=2.5, ard_num_dims=4)
    kernel = ScaleKernel(base_kernel)
    likelihood = PairwiseLogitLikelihood(noise_likelihood=noise_likelihood)
    model = PairwiseGP(
            train_x,
            train_comp,
            covar_module=kernel,
            likelihood=likelihood,
            jitter=1e-4
        )   
    
    # Model fitting
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    success = False
    for attempt in range(3):
        try:
            fit_gpytorch_mll(mll)
            success = True
            break
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            torch.manual_seed(torch.randint(0, 10000, (1,)).item())

    if not success:
        print("❌ Fit failed, returning untrained model.")

    new_l = model.covar_module.base_kernel.lengthscale
    new_var = model.covar_module.outputscale

    return train_x, train_comp, model, mll, visited_pairs, new_l, new_var, likelihood

