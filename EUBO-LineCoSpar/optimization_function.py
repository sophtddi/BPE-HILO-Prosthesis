"""
This file contains specific functions used during the optimization phase of the experiment in the optimization.py file.

- initial_acquisition(xs):
Randomly selects two configurations from a grid of possible configurations to start the optimization process.

- acquisition(xs, model, current_best, visited_pairs, max_attempts):
Selects a new challenger configuration to compare against the current best configuration, ensuring the pair hasn't been visited before.

- cofeedback(xs, current_best, ub, lb, n_grid, pending_coactive_pt): 
Handles user feedback to adjust the current best configuration based on specific parameter changes.

- early_stopping(previous_best, current_best, n_min, n_stop, no_improvement_counter, i):
Defines convergence criteria

- get_random_direction(dim):
Generates a random direction vector for exploration.

- points_along_line(best_point, direction, grid_points, visited_points):
Selects candidate points along a line in a given direction.
"""

import torch
import numpy as np

from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from core_functions import translate_param
from config import lb, ub, n_grid, radius


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


def initial_acquisition(xs):
    """
    When starting the optimization, we choose 2 random configurations to compare
    Args:
        xs: grid of all the possible configuration

    Returns:
        intial pair: pair of 2 configurations
    """
    indices = np.random.choice(len(xs), size=2, replace=False) # randomly select 2 indices 
    point1 = xs[indices[0]]
    point2 = xs[indices[1]]
    initial_pair = torch.stack([point1, point2])    
    # print(f"in initial_acq pair is {initial_pair}")

    return initial_pair

def acquisition(xs, model, current_best, visited_pairs, max_attempts=(50)):
    """"
    Select a new challenger based on a strategy and compare it to the current best
    Ensure we don't propose a previously visited pair

    Args:
        xs: grid of all possible configuration to choose from
        model: model
        current_best: current prefered configuration
        visited_pairs: list of visited pairs
        max_attempts: number of maximum comparison allowed to find a new pair 

    Returns:
        next_pair: [current_best, challenger]
        policy: policy used to find the new challenger
    """
    
    def is_pair_visited(pair, tol=1e-2):     # check if a pair has been visited before
        for visited_pair in visited_pairs:             # Check both orderings (a,b) and (b,a)
            if (torch.allclose(pair[0], visited_pair[0], atol=tol) and 
                torch.allclose(pair[1], visited_pair[1], atol=tol)) or \
               (torch.allclose(pair[0], visited_pair[1], atol=tol) and 
                torch.allclose(pair[1], visited_pair[0], atol=tol)):
                return True
        return False
    
    ## Find a  challenger to the current best point 
    attempts = 0
    while attempts < max_attempts: # the challenger and the best point should not form an already visited pair
        # build the policy based on the acquistion function, current model, and current best point
        policy = AnalyticExpectedUtilityOfBestOption(   
            pref_model=model,
            previous_winner=current_best,
        )
        # use the policy to find the next challenger among the xs
        challenger, _ = optimize_acqf_discrete(
            policy,
            choices=xs,
            q=1,
        )
    
        next_pair = torch.cat([current_best, challenger], dim=0) # build the pair
        if not is_pair_visited(next_pair):  # Check if this pair has been visited before
            return next_pair, policy        # if not, this is the pair to compare   
        
        attempts += 1
        if attempts >= max_attempts:                # if maximum attempts reached
            idx = torch.randperm(len(xs))[0]        #r randomly select the challenger                 
            challenger = xs[idx:idx+1]
            next_pair = torch.cat([current_best, challenger], dim=0)
            return next_pair, policy

def cofeedback(xs, current_best, ub, lb, n_grid, pending_coactive_pt):
    while True:
            feedback = input("\nCofeedback? (y, n): ")
            if feedback == 'y':
                param = input("Which param? (0=t1, 1=k1, 2=theta1 or 3=k2): ") 
                if param in ['0', '1', '2', '3']:
                    direction = input("Which direction? (-1, 1)")
                    if direction in ['-1', '1']:
                        max_param = int(param)
                        max_direction = int(direction)
                        current_point_tensor = current_best.clone().detach().requires_grad_(True)
                        coactive_pt = current_point_tensor.detach().cpu().numpy().ravel().copy()
                        coactive_pt[max_param] = coactive_pt[max_param] + max_direction*(ub-lb)/(n_grid-1)
                        coactive_pt_tensor = torch.tensor(coactive_pt, dtype=torch.float32, device=device).view(1, -1)
                                            
                        found_match = False
                        for x in xs:
                            if torch.all(torch.isclose(x, coactive_pt_tensor, atol=1e-2)):
                                found_match = True
                                break

                        if not found_match: 
                            pending_coactive_pt = None
                            print('The cofeedback is out of grid so will use normal acquisition')
                            break
                        else:
                            pending_coactive_pt = coactive_pt_tensor
                            print("The proposed coactive point is", translate_param(pending_coactive_pt))
                            break   
            elif feedback == 'n':
                break
            else:
                print('Select y or n')
    return pending_coactive_pt

def early_stopping(previous_best, current_best, n_min, n_stop, no_improvement_counter, i): 
    if torch.equal(previous_best, current_best) and i > n_min:
        no_improvement_counter += 1
    else:
        no_improvement_counter = 0

    if no_improvement_counter >= n_stop:
        print(f"\nStopping after {n_stop} iterations without improvement.")
        return True, no_improvement_counter

    return False, no_improvement_counter

def get_random_direction(dim, device=None):
    """
    Generate a vector in a random direction to select a line of exploration

    Args:
        dim: dimension of the problem
    Returns:
        direction: vector
    """
    direction = torch.randn(dim, device=device) # generate a random 4D-vector
    direction /= torch.norm(direction)          # normalize it 
    return direction

def points_along_line(best_point, direction, grid_points, visited_points=None, lb=lb, ub=ub, n_grid=n_grid, radius=radius):
    """
    Select the points among which we are going to select the challenger.
    The points are considered if they are on the line (with a small radius)
    passing through the current best point in a given direction,
    or if they have already been visited.

    Args:
        best_point: current best point, the line passes through this point
        direction: a randomly generated direction vector
        grid_points: all possible configurations in the search space
        visited_points: (optional) previously visited configurations
        lb: lower bound of the search space
        ub: upper bound of the search space
        n_grid: number of grid points per dimension
        radius: tolerance factor for how far points can be from the line

    Returns:
        combined_points: candidate points among which to select the next challenger
    """
    grid_spacing = (ub - lb) / (n_grid - 1)     # compute spacing between grid points in each dimension
    tolerance = grid_spacing * radius           # tolerance radius around the line

    vectors_to_grid = grid_points - best_point  # vector from best_point to each grid point
    projections = torch.sum(vectors_to_grid * direction, dim=1)                             # project those vectors onto the direction vector
    projections_on_line = best_point + projections.unsqueeze(1) * direction.unsqueeze(0)    # projected points back onto the line
    distances_to_line = torch.norm(grid_points - projections_on_line, dim=1)                # distance from each grid point to the line

    # Mask: select grid points close enough to the line
    line_points_mask = distances_to_line <= tolerance
    line_points = grid_points[line_points_mask]
    projections_selected = projections[line_points_mask]

    # Sort line points along the projection direction
    if len(line_points) > 0:
        sorted_indices = torch.argsort(projections_selected)
        line_points = line_points[sorted_indices]

    # If no visited points provided, return only the line points
    if visited_points is None:
        return line_points

    # Otherwise, add previously visited points to the candidate set
    combined_points = line_points
    for point in visited_points:
        is_duplicate = False
        if len(combined_points) > 0:
            distances = torch.norm(combined_points - point, dim=1)
            is_duplicate = torch.min(distances) < 1e-6  # Avoid adding duplicates

        if not is_duplicate:
            point_reshaped = point.view(1, -1)
            combined_points = torch.cat([point_reshaped, combined_points], dim=0)

    return combined_points
