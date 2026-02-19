"""
This file contains the pipeline to run the exploration that is the initialisation of the BO model.
It consists of asking user preference over predefined pairs_to_test defined in the config.py file.
The model is based on a PairwiseGP model built over pairwise comparison, it recquires:
    - train_x : that are the tested configuration normalized between 0 and 1 (for ex train_x = [1, 0.2, 0.4])
    - train_comp: that are the user preference to the proposed pairs in the shape [i,j] 
                    where i,j are the indices of the 2 configuration in train_x
                    with i being prefered to j (for ex if 0.4 is prefered over 1 then train_comp = [2, 0])

The posterior of the latent function is computed from a Gaussian Process with a RBF kernel scaled by an outputscale factor and the likelihood.
In practice, it is computed by a Laplace Approximation by optimizing the PairwiseLaplaceMarginalLogLikelihood.
"""

import numpy as np 
import os

import warnings 
from botorch.exceptions.warnings import OptimizationWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=OptimizationWarning)
from botorch import settings
settings.debug(True)

from core_functions import *            # import core functions
from exploration_function import *      # import exploration functions

''' PROBLEM DEFINITION '''      
seed = 130                                                  # fixing randomness
torch.manual_seed(seed)
np.random.seed(seed)
from config import all_config, lengthscale, pairs_to_test   # import all possible configuration, initial lengthscale and the pairs to test  
from config import save_dir, file_path                      # for saving data purpose
os.makedirs(save_dir, exist_ok=True)

""" EXPLORATION """
print('\nEXPLORATION')
train_x, train_comp, model, mll, visited_pairs, lengthscale, outputscale = problem_exploration(xs=all_config, pairs_to_test=pairs_to_test, l=lengthscale) # run the exploration

translate_train_x = [translate_param(x) for x in train_x]           # scale back the configurations to the real range of values
print(f'Exploration with:\n {torch.tensor(translate_train_x)}')
print(f'l is {lengthscale}, outpuscale is {outputscale}') 

save_model(None, None, train_x, train_comp, model, mll, lengthscale, outputscale, visited_pairs, save_dir, 'explo') # save model

print(f"Model saved in '{save_dir}'")

# for results analysis purpose
posterior = model.posterior(train_x)            # compute the posterior from the model over the visited configuration
mean = posterior.mean.squeeze(-1)               # mean estimated value for every visited configuration
variance = posterior.variance.squeeze(-1)       # variance for every visited configuration 
print(mean)
print(variance)



