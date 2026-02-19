"""
This file contains the pipeline to run exploration.
In the exploration session, we start by loading the model built after initial design.
At every iteration, we use EUBO as acquisition function to choose the pair of points to propose to the user. 
The proposed point shouldn't be to close to each other in euclidian distance, and should form a new pair.
We ask  the user preference between the two points, and update the model.

Note: 
    by default the points are considered in their normalized version (ie between 0 an 1) to train the model
    therefore they need to be scaled back to their range of values afterward for real life implementation to the prosthesis
"""

import os
import warnings 
from botorch.exceptions.warnings import OptimizationWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=OptimizationWarning)

from exploration_function import * # import of optimization functions
from core_function import *        # import of core functions
from ID_function import *  # import of exploration functions
from config import *                    # import all the experiment setup



# ===================== PROBLEM DEFINITION =====================
test_id = 3                              # id of the trial (for saving data purpose)
os.makedirs(save_dir, exist_ok=True)    # create the folder to save data
seed = 13                              # to control randomness (each trial should be a different seed)
torch.manual_seed(seed)
np.random.seed(seed)
# [57.8, 0.49, -46.2, 0.43]

# [47, 1.5, -45, 0.5]


# ===================== INITIAL DESIGN - LOADING DATA =====================
print('\nINITIAL DESIGN - LOADING DATA')

try:
    # Load initial design data
    model, likelihood, checkpoint = retrain_model(file_path, noise_likelihood=noise_likelihood)
    train_x = checkpoint['train_x']                 # load points visisted during exploration
    visited_pairs = checkpoint['visited_pairs']     # load pair comparison visisted during exploration
    train_comp = checkpoint['train_comp']           # load user preference over comparison visisted during exploration
    mll = checkpoint['mll']                         # marginal loglikelihood of the model constructed during exploration
    lengthscale = model.covar_module.base_kernel.lengthscale                            # lengthscale of the trained model after exploration         
    var = model.covar_module.outputscale                                                # outputscale of the trained model after exploration    
    print(f'Data loaded with success! (lengthscale: {lengthscale}, var: {var}) ')
    
except Exception as e:     # if loading failed, run exploration 
    print(f"Error in loading data: {e}. Running initial design manually again.")
    train_x, train_comp, model, mll, visited_pairs, lengthscale, var, likelihood = initial_design(pairs_to_test=pairs_to_test, dim=dim, noise_likelihood=noise_likelihood)

translated_train_x = torch.from_numpy(np.array([translate_param(x.numpy()) for x in train_x])).double()
print('Initial design with:', len(translated_train_x))


# ===================== EXPLORATION =====================
print('\nEXPLORATION')
pref_boolean = True
stored_pair = None
stored_third = None
pending_fallback = False
no_pref_count = 0
estimate_history = []  # Track estimates for convergence check

for i in range(n_queries):
    print(f"\nExploration iteration {i}")

    if pref_boolean:
        next_pair, third_point, policy = acquisition(model=model,dim=dim,visited_pairs=visited_pairs)
        print(f"Third point is {third_point}")
        stored_pair = next_pair.clone()
        # stored_third = third_point.clone()
        stored_third = third_point.clone().squeeze(0) if third_point.dim() > 1 else third_point.clone()

        pending_fallback = False

    else:
        print(">No preference before: Fallback comparison triggered")
        next_pair = torch.stack([stored_pair[0], stored_third])
        pending_fallback = True   
        pref_boolean = True       

    print(f"Option 1: {next_pair[0]} vs Option 2: {next_pair[1]}")
    next_comp, current_best, current_unliked = user_preference(pair=next_pair)

    if (next_comp == -1).all():
        print("User has NO preference → no data added")
        pref_boolean = False
        # increment counter of consecutive 'no preference'
        no_pref_count += 1
        # after two consecutive no-preference answers, diversify by creating a fourth random point
        if no_pref_count >= 2:
            print("Two consecutive 'no preference' responses: generating a fourth challenger to diversify.")
            # create a new random challenger in normalized space
            fourth_point = torch.rand(dim)
            # replace stored_third so the fallback will compare stored_pair[0] vs this new fourth point
            stored_third = fourth_point.clone()
            # ensure next iteration will use the fallback with the new fourth point
            pending_fallback = True
        # print(train_x, train_comp)
        save_model(best_config=i_pref_estimate, option1=next_pair[0], option2=next_pair[1], pref=None,
                train_x=train_x, train_comp=train_comp, visited_pairs=visited_pairs,
                model=model, likelihood=likelihood, mll=mll,
                save_dir=save_dir, test_id=test_id, i=f'_it{i}_')
        continue

    print("User provided a preference")
    # reset counter when user gives a real preference
    no_pref_count = 0
    train_x, train_comp = append_data(x_next=next_pair, x_train=train_x, comp_next=next_comp, comp_train=train_comp)
    print(f"Added main pair with comp {next_comp}")
    # Record the proposed pair as visited so acquisition won't propose it again
    try:
        # store a CPU/double clone to keep types consistent with other code
        visited_pairs.append(next_pair.clone().detach().cpu().double())
    except Exception:
        visited_pairs.append(next_pair.clone())

    if pending_fallback:
        print(f"Applying deduction from fallback with next_comp is {next_comp.tolist()}")

        if next_comp[0].tolist() == [0, 1]:             # stored_pair[0] ≻ stored_third  → deduce stored_pair[1] ≻ stored_third
            print("Deduction: stored_pair[1] ≻ stored_third")
            pair_BC = torch.stack([stored_pair[1], stored_third])
            comp_BC = torch.tensor([[0, 1]])
            train_x, train_comp = append_data(x_next=pair_BC, x_train=train_x, comp_next=comp_BC, comp_train=train_comp)
            visited_pairs.append(pair_BC)

        elif next_comp[0].tolist() == [1, 0]:           # stored_third ≻ stored_pair[0]  → deduce stored_third ≻ stored_pair[1]
            print("Deduction: stored_third ≻ stored_pair[1]")
            pair_CB = torch.stack([stored_third, stored_pair[1]])
            comp_CB = torch.tensor([[0, 1]])
            train_x, train_comp = append_data(x_next=pair_CB, x_train=train_x, comp_next=comp_CB, comp_train=train_comp)
            visited_pairs.append(pair_CB)

        pending_fallback = False
    # print(train_x, train_comp)

    # Update model
    model, mll, likelihood = update_model(model=model, mll=mll, likelihood=likelihood, train_x=train_x, train_comp=train_comp, noise_likelihood=noise_likelihood)                             # update the model

    # Save model at iteration i
    i_pref_estimate = max_utility_estimate(model)                               
    save_model(best_config=i_pref_estimate, option1=next_pair[0], option2=next_pair[1], pref=current_best,
                train_x=train_x, train_comp=train_comp, visited_pairs=visited_pairs,
                model=model, likelihood=likelihood, mll=mll,
                save_dir=save_dir, test_id=test_id, i=f'_it{i}_')
    
    lengthscale = model.covar_module.base_kernel.lengthscale                                                            # lengthscale of the model
    var = model.covar_module.outputscale
    print(f'Updated model hyperparameters -> lengthscale: {lengthscale}, var: {var})')
    print(f'Estimated preference:  {translate_param(i_pref_estimate).tolist()}')
    
    # Track estimate for convergence check
    estimate_history.append(i_pref_estimate.clone().detach())
    
    # Check if estimates have converged (stagnated for n_iterations with < 10% change per dimension)
    if check_convergence(estimate_history, dim=dim, n_iterations=5, threshold=0.1) and i >= 15:
        print(f"\n✅ Convergence detected: estimates have stagnated over last 5 iterations.")
        print(f"Stopping exploration early at iteration {i}.")
        break

# ===================== FINAL ESTIMATE =====================
final_pref_estimate = max_utility_estimate(model)
save_model(best_config=final_pref_estimate.tolist(), option1=None, option2=None, pref=None,
                train_x=train_x, train_comp=train_comp,  visited_pairs=visited_pairs,
                  model=model, likelihood=likelihood, mll=mll,
                  save_dir=save_dir, test_id=test_id, i='_final_model')
print(f"Model estimate of preference: {translate_param(final_pref_estimate).tolist()}")

