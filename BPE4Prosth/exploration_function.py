"""
This file contains specific functions used during the exploration phase of the experiment in the exploration.py file.

- acquisition(model, dim, visited_pairs=None, max_attempts=5, min_relative_diff=0.1)
Selects a pair of configuration based on EUBO, ensuring the pair hasn't been visited before and the points are not too close.

- max_utility_estimate(model, dim=dim):
Give the estimated preference based on the model
"""

import torch
import numpy as np

from botorch.optim import optimize_acqf
from botorch.acquisition.preference import qExpectedUtilityOfBestOption
from botorch.acquisition.analytic import PosteriorMean
from config import dim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def acquisition(model, dim, visited_pairs=None, max_attempts=5, min_relative_diff=0.1):
    """
    Select a new challenger based on an acquisition strategy and compare it to the current best.
    Ensures:
      - The new pair is not too close (min_relative_diff)
      - The pair has not already been visited (order-invariant check)
      - If no valid pair is found after max_attempts, returns a random pair.

    Args:
        model: trained preference model
        dim (int): dimensionality of each point
        visited_pairs (list[torch.Tensor]): list of tensors, each of shape (2, dim)
        max_attempts (int): max attempts to find a valid pair
        min_relative_diff (float): minimum relative difference on at least one coordinate

    Returns:
        next_pair (torch.Tensor): tensor of shape (2, dim)
        policy: acquisition policy used
    """
    policy = qExpectedUtilityOfBestOption(pref_model=model)
    if visited_pairs is None:
        visited_pairs = []

    def is_same_pair(p1, p2, tol=1e-1):
        """
        Check if two pairs (2, dim) represent the same configuration, order invariant.
        """
        # same order or reversed
        return (torch.allclose(p1[0], p2[0], atol=tol) and torch.allclose(p1[1], p2[1], atol=tol)) or \
               (torch.allclose(p1[0], p2[1], atol=tol) and torch.allclose(p1[1], p2[0], atol=tol))

    for attempt in range(max_attempts):
        print(f"Attempt {attempt + 1}/{max_attempts}")

        # Generate candidate triplet (q = 3)
        next_triplet, _ = optimize_acqf(
            policy,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]),
            q=3,                     
            num_restarts=30,
            raw_samples=512,
        )

        # On garde la logique "next_pair" = les 2 premiers points
        point1, point2 = next_triplet[0], next_triplet[1]
        third_point = next_triplet[2]    

        # # Check distance only between the first two points
        # rel_diff = torch.abs(point1 - point2)
        # if not torch.any(rel_diff >= min_relative_diff):
        #     continue             # Points too close → retry

        # Check if already visited (only for the pair)
        next_pair = next_triplet[:2]       # tensor of shape (2, dim)

        already_visited = any(is_same_pair(next_pair, vp) for vp in visited_pairs)
        if already_visited:
            print("Pair already visited — retrying.")
            continue

        # Valid pair found
        print("✅ New valid pair found.")
        return next_pair, third_point, policy


    # If all attempts failed → return [point1, point2 + noise]
    print("⚠️ No new valid pair found after max attempts. Returning random pair.")
    new_point = torch.clamp(point2 + 0.1 * torch.rand((1, dim)), max=1.0)
    third_point = torch.clamp(point2 + 0.3 * torch.rand((1, dim)), max=1.0)
    random_pair = torch.cat([new_point, point1.unsqueeze(0)], dim=0)
    return random_pair, third_point, policy

def max_utility_estimate(model, dim=dim):
    """
    Give the estimated preference based on the model
    """
    mean_acqf = PosteriorMean(model)

    best_estimate, _ = optimize_acqf(
        mean_acqf,
        bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]),
        q=1,
        num_restarts=30,
        raw_samples=512,
    )

    return best_estimate


def check_convergence(estimate_history, dim, n_iterations=5, threshold=0.1):
    """
    Check if the estimated preference has converged (stagnated) over n_iterations.
    
    Convergence is defined as: for each dimension d, the relative change between
    any two consecutive estimates in the last n_iterations is < threshold (10% by default).
    
    Args:
        estimate_history (list[torch.Tensor]): list of estimates from past iterations,
                                                each of shape (1, dim) or (dim,)
        dim (int): dimensionality
        n_iterations (int): number of iterations to check for stagnation (default 5)
        threshold (float): maximum relative change per dimension to consider stagnated (default 0.1 = 10%)
    
    Returns:
        converged (bool): True if all dimensions have changed < threshold for all
                         consecutive pairs in the last n_iterations; False otherwise
    """
    if len(estimate_history) < n_iterations:
        # Not enough history yet
        return False
    
    # Get last n_iterations estimates
    recent = estimate_history[-n_iterations:]
    
    # Flatten to ensure shape (n, dim)
    recent_flat = []
    for est in recent:
        if est.dim() == 2:
            recent_flat.append(est.squeeze(0))
        else:
            recent_flat.append(est)
    
    # Check relative change between all consecutive pairs
    for i in range(len(recent_flat) - 1):
        curr = recent_flat[i]
        next_est = recent_flat[i + 1]
        
        for d in range(dim):
            # Avoid division by zero: if both are very small, consider no change
            if abs(curr[d].item()) < 1e-6 and abs(next_est[d].item()) < 1e-6:
                rel_change = 0.0
            else:
                denominator = max(abs(curr[d].item()), 1e-6)
                rel_change = abs((next_est[d].item() - curr[d].item()) / denominator)
            
            # If any dimension changes >= threshold, not converged
            if rel_change >= threshold:
                return False
    
    # All dimensions in all consecutive pairs have small changes
    return True