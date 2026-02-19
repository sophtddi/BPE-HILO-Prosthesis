"""
This file contains the pipeline to run the inital design for the BO model.
It consists of asking user preference over random pairs, called pairs_to_test defined in the config.py file.
The model is based on a PairwiseGP model built over pairwise comparison, it recquires:
    - train_x : that are the tested configuration normalized between 0 and 1 (for ex train_x = [1, 0.2, 0.4, 1])
    - train_comp: that are the user preference to the proposed pairs in the shape [i,j] 
                    where i,j are the indices of the 2 configuration in train_x
                    with i being prefered to j (for ex if 0.4 is prefered over 1 then train_comp = [2, 0])

The posterior of the latent function is computed from a Gaussian Process with a RBF kernel scaled by an outputscale factor and the likelihood.
In practice, it is computed by a Laplace Approximation by optimizing the PairwiseLaplaceMarginalLogLikelihood.
"""

import numpy as np 
import os
import botorch
import torch
import warnings 
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.set_default_dtype(torch.double)

from core_function import *             # import core functions
from ID_function import *               # import exploration functions
# # [57.8, 0.49, -46.2, 0.43]


# [47, 1.5, -45, 0.5]
# ===================== PROBLEM DEFINITION =====================
seed = 131                                                      # fixing randomness
torch.manual_seed(seed)
np.random.seed(seed)
from config import pairs_to_test, save_dir, noise_likelihood, dim    # import pairs to test  
os.makedirs(save_dir, exist_ok=True)

# ===================== RUN INITIAL DESIGN =====================
print('\nINITIAL DESIGN')
train_x, train_comp, model, mll, visited_pairs, lengthscale, outputscale, likelihood = initial_design(pairs_to_test=pairs_to_test, dim=dim, noise_likelihood=noise_likelihood) # run the exploration


# ===================== DISPLAY INITIAL DESIGN =====================
translated_train_x = torch.from_numpy(np.array([translate_param(x.numpy()) for x in train_x])).double()
print(f'Exploration with:\n {torch.tensor(translated_train_x)}')
print(f'l is {lengthscale}, outpuscale is {outputscale}') 


# ===================== SAVE INITIAL DESIGN =====================
save_model(best_config=None, option1=None, option2=None, pref=None,
                train_x=train_x, train_comp=train_comp, visited_pairs=visited_pairs,
                model=model, likelihood=likelihood, mll=mll,
                save_dir=save_dir, test_id='explo')
print(f"Model saved in '{save_dir}'")
