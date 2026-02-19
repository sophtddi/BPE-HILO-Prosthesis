"""
This file contains the pipeline to run one trial of optimization followed by the validation session.
In the optimization session, we start by loading the model built after exploration.
Then we start optimization with a random pair of 2 configurations and ask user preference.
From this preference, we then update the model. 
The model is then used to choose the next challenger on a list of points containing the points that 
are on a line passing through the current_best in a random direction and the already visited points.
The challenger is chose among those points by maximising the Expected Utility of Best Option (EUBO).
We then ask again the user preference between his current_best and the challenger, and update the model again.
And so on until convergence, convergence being define has keeping the same points as current_best for
a given number of iterations (n_iterations).
We then enter validation phase where the user is proposed a pair containing the converged configuration
and a random selected challenger in random order.

Note: 
    by default the points are considered in their normalized version (ie between 0 an 1) to train the model
    therefore they need to be scaled back to their range of values afterward for real life implementation to the prostesis
"""

import os

import warnings 
from botorch.exceptions.warnings import OptimizationWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=OptimizationWarning)
from botorch import settings
settings.debug(True)

from optimization_function import * # import of optimization functions
from core_functions import *        # import of core functions
from exploration_function import *  # import of exploration functions
from validation_function import *   # import of valdiation functions


'''PROBLEM DEFINITION'''
test_id = 3                             # id of the trial (for saving data purpose)
from config import *                    # import all the experiment setup
os.makedirs(save_dir, exist_ok=True)    # create the folder to save data

seed = 10                               # to control randomness (each trial should be a different seed)
torch.manual_seed(seed)
np.random.seed(seed)

# ===================== EXPLORATION PART - LOADING DATA =====================
print('\nEXPLORATION PART - LOADING DATA')

try:
    # Load exploration data
    checkpoint = torch.load(file_path)              # download saved exploration data from exploration trial
    train_x = checkpoint['train_x']                 # load points visisted during exploration
    visited_pairs = checkpoint['visited_pairs']     # load pair comparison visisted during exploration
    train_comp = checkpoint['train_comp']           # load user preference over comparison visisted during exploration
    mll = checkpoint['mll']                         # marginal loglikelihood of the model constructed during exploration
    
    # Rebuild the model based on a RBF kernel k(x,x') = outputscale² * exp(-||x-x'||²/(2*lengthscale**²))
    base_kernel = RBFKernel(lengthscale_constraint=Interval(*length_bounds))            # RBF kernel constrained within [minimal grid space, maximal grid space] 
    kernel = ScaleKernel(base_kernel, outputscale_constraint=Interval(*out_bounds))     # ScaleKernel allows to scale the kernel by outputscale that is constrained too
    model = PairwiseGP(train_x, train_comp, covar_module=kernel, jitter=1e-2)           # model is a PairwiseGP
    model.load_state_dict(checkpoint['model_state_dict'])                               # load the model hyperparamters
    lengthscale = model.covar_module.base_kernel.lengthscale                            # lengthscale of the trained model after exploration         
    var = model.covar_module.outputscale                                                # outputscale of the trained model after exploration    
    print(f'Data loaded with success! (lengthscale: {lengthscale}, var: {var}) ')
    
except Exception as e:
    # if loading failed, run exploration again
    print(f"Error in loading data: {e}. Running exploration manually again.")
    train_x, train_comp, model, mll, visited_pairs, lengthscale = problem_exploration(xs=all_config, pairs_to_test=pairs_to_test, l=lengthscale)

print('Exploration with', train_x)

# ===================== OPTIMIZATION INITIALISATION =====================
print('\nOPTIMIZATION INITIALISATION')
no_improvement_counter = 0   # follow the number of iteration the user keep the same configuration as current_best (for convergence analysis)

initial_pair = initial_acquisition(xs=all_config)                   # the optimization starts with a random pair of configuration among all possibilities
initial_comp, initial_best = user_preference(pair=initial_pair)     # ask user preference between the 2 initial configurations

train_x, train_comp = append_data(x_next=initial_pair, x_train=train_x, comp_next=initial_comp, comp_train=train_comp)      # add the data 
model, mll = update_model(model=model, mll=mll, train_x=train_x, train_comp=train_comp)                                     # update the model
lengthscale = model.covar_module.base_kernel.lengthscale                                                                    # lengthscale of the new model
var = model.covar_module.outputscale                                                                                        # outputscale of the new model
print(f'Updated model hyperparameters -> lengthscale: {lengthscale}, var: {var})')


current_best = initial_best                                                                 # update user preference
print(f"Preference so far: {(current_best)} -> {translate_param(current_best)} ")
save_model(translate_param(current_best), None, train_x, train_comp, model, mll, lengthscale, var, visited_pairs, save_dir, test_id, '_init_') # save model

# ===================== OPTIMIZATION LOOP =====================
print('\nOPTIMIZATION LOOP')
for i in range(n_queries):                  # for a maximum of n_queries, try to converge to user preference
    print('\nOptimization iteration', i)
    previous_best = current_best            # previous_best keeps in memory the prefered configuration of previous iteration

    # Acquisition: to find the challenger to the current_best
    if cofeedback_pt is None:       # if user didn't give any feedback via cofeedback_pt
        # consider the points that are on a line in a random direction passing trough current_best and already visited points as potential candidates -> line_to_sample
        random_direction = get_random_direction(dim=dim)        
        line_to_sample = points_along_line(best_point=current_best, direction=random_direction, grid_points=all_config, visited_points=train_x, lb=lb, ub=ub, n_grid=n_grid)
        # choose the point (challenger) among line_to_sample that maximises the acquisition function and make a pair -> next_pair = [current_best, challenger]
        next_pair, policy = acquisition(xs=line_to_sample, model=model, current_best=current_best, visited_pairs=visited_pairs) 
    else:
        challenger = cofeedback_pt # if user gave a feedback via cofeedback_pt
        next_pair = torch.cat([current_best, challenger], dim=0)    # then the pair to compare is next_pair = [current_best, challenger]
        cofeedback_pt = None                                        # set the cofeeback input back to None again
    
    # Display of the parameters to send them to the prosthesis 
    best_translate = translate_param(next_pair[0])          # scale back current_best from its normalized version to its real parameters range of values
    challenger_translate = translate_param(next_pair[1])    # scale back challenger from its normalized version to its real parameters range of values
    print(f"Previous best: {best_translate} vs Challenger: {challenger_translate}")

    # Ask for preference
    next_comp, current_best = user_preference(pair=next_pair)

    # Update data and model
    visited_pairs.append(next_pair)                                                                                     # store visited pairs
    train_x, train_comp = append_data(x_next=next_pair, x_train=train_x, comp_next=next_comp, comp_train=train_comp)    # update the training data
    model, mll = update_model(model=model, mll=mll, train_x=train_x, train_comp=train_comp)                             # update the model
    lengthscale = model.covar_module.base_kernel.lengthscale                                                            # lengthscale of the model
    var = model.covar_module.outputscale                                                                                # outputscale of the model
    save_model(best_translate, challenger_translate, train_x, train_comp, model, mll, lengthscale, var, visited_pairs, save_dir, test_id, f'_it{i}_')  # save data
    print(f'Updated model hyperparameters -> lengthscale: {lengthscale}, var: {var})')

    # Cofeedback: if the user want to give some feedback
    if boolean_cofeedback and cofeedback_pt is None: 
        cofeedback_pt = cofeedback(xs=all_config, current_best=current_best, lb=lb, ub=ub, n_grid=n_grid, pending_coactive_pt=cofeedback_pt) # cofeedback_pt will be the next challenger

    # Check stop condition: after n_min iterations, if a point is current_best for n_stop iterations, we have convergence
    stop, no_improvement_counter = early_stopping(previous_best, current_best, n_min, n_stop, no_improvement_counter, i)
    if stop:
        break
    
# Display 
best_translate = translate_param(current_best)  # scale back current_best to parameters range of values
posterior = model.posterior(train_x)            # model posterior on visited points 
mean = posterior.mean.squeeze(-1)               # model mean estimation on visited points
variance = posterior.variance.squeeze(-1)       # model variance estimation on visited points
print([translate_param(x) for x in train_x])
print(mean)
print(variance)



# ===================== VALIDATION TESTS =====================
print('\nVALIDATION')

for i in range(n_verif):                                        # for n_verif iterations
    idx = torch.randperm(len(all_config))[0]            
    challenger = all_config[idx:idx+1]                          # choose a random challenger
    choice = validation_test(best_translate, challenger, i)     # ask the user its preference between current_best and challenger in a random order
    print(f'User chose option {choice}, the true preference was {best_translate}')

    save_model(best_translate, translate_param(challenger), train_x, train_comp, model, mll, lengthscale, var, visited_pairs,save_dir, test_id, f'_validation_{i}_')

