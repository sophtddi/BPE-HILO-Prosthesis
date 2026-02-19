""".
This file contains functions and parameters for generating grid points and defining the setup for exploration, optimization and validation processes. 
"""

import os
import torch
import numpy as np
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

torch.set_default_dtype(torch.double)
np.random.seed(1)

""" SAVING PARAMETERS """
save_dir = 'Michel_912'        # directory to save the model and data
explo_file = 'explo_data.pt'
file_path = os.path.join(save_dir, explo_file)

""" GRID DEFINITION """
dim = 4             # dimension of the problem
lower_bounds = [40, 0.4, -60, 0.35]
upper_bounds = [60, 1.8, -40, 0.6]
noise_likelihood = 0.7

""" MODEL HYPERPARAMETERS """
# RBF kernel k(x,x') = outpuscale² * exp(-||x-x'||²/(2*lengthscale²)

"""" EXPLORATION PARAMETERS """

def make_pairs_from_pool(pool):
    """
    Given an array of points of shape (2n, d), split into n pairs by random matching.
    """
    n = pool.shape[0] // 2
    idx = np.random.permutation(pool.shape[0])
    A = pool[idx[:n]]
    B = pool[idx[n:2*n]]
    return np.stack([A, B], axis=1)


def max_distance_pairs_from_pool(pool):
    """
    Build pairs maximizing total distance between the two elements.
    pool must contain exactly 2*n points.
    """
    n = pool.shape[0] // 2
    A = pool[:n]
    B = pool[n:2*n]
    D = cdist(A, B)
    # maximize sum of distances -> minimize (-D)
    row_ind, col_ind = linear_sum_assignment(-D)
    return np.stack([A[row_ind], B[col_ind]], axis=1)


def max_distance_per_dimension_pairs_from_pool(pool, dim=4, rng=None):
    """
    Build pairs that maximize distance for each dimension independently.
    For each dimension, pair the point with smallest value with point with largest value.
    This ensures high variance in each dimension.
    
    Args:
        pool: array of shape (2*n, dim)
        dim: number of dimensions
        rng: numpy.random.Generator instance for shuffling
    
    Returns:
        pairs: array of shape (n, 2, dim) where each pair maximizes distance per dimension
    """
    # Improved algorithm:
    # - Split pool into A (first n) and B (last n)
    # - For each dimension d compute an assignment that maximizes absolute difference
    #   along that single dimension using linear_sum_assignment on |A[:,d]-B[:,d]|.
    # - Collect candidate pairs from all dimensions, sort them by their per-dim
    #   distance (descending) and greedily pick pairs that do not reuse points
    #   until we obtain n disjoint pairs. If not enough candidates remain, fall
    #   back to a global max-distance matching on the remaining points.

    n = pool.shape[0] // 2
    if pool.shape[0] != 2 * n:
        raise ValueError("pool must contain exactly 2*n points")

    A = pool[:n]
    B = pool[n:2 * n]

    # Candidate pairs across dimensions: list of tuples (a_idx, b_idx, dim, dist)
    candidates = []
    for d in range(dim):
        # distance matrix for this single dimension
        Dd = np.abs(A[:, d][:, None] - B[:, d][None, :])
        # maximize -> linear_sum_assignment on -Dd
        row_ind, col_ind = linear_sum_assignment(-Dd)
        for r, c in zip(row_ind, col_ind):
            candidates.append((r, c, d, float(Dd[r, c])))

    # sort candidates by distance desc
    candidates.sort(key=lambda t: -t[3])

    used_A = np.zeros(n, dtype=bool)
    used_B = np.zeros(n, dtype=bool)
    selected_pairs = []

    for a_idx, b_idx, d_idx, dist in candidates:
        if not used_A[a_idx] and not used_B[b_idx]:
            used_A[a_idx] = True
            used_B[b_idx] = True
            selected_pairs.append([A[a_idx], B[b_idx]])
            if len(selected_pairs) == n:
                break

    # If not enough disjoint pairs were found, fall back to global max on remaining
    if len(selected_pairs) < n:
        # collect remaining unused points
        rem_A_idx = np.where(~used_A)[0]
        rem_B_idx = np.where(~used_B)[0]
        if rem_A_idx.size > 0 and rem_B_idx.size > 0:
            rem_A = A[rem_A_idx]
            rem_B = B[rem_B_idx]
            Drem = cdist(rem_A, rem_B)
            row_rem, col_rem = linear_sum_assignment(-Drem)
            for r, c in zip(row_rem, col_rem):
                selected_pairs.append([rem_A[r], rem_B[c]])
                if len(selected_pairs) == n:
                    break

    pairs = np.array(selected_pairs)
    # Ensure shape is (n,2,dim)
    if pairs.shape[0] != n:
        raise RuntimeError(f"Unable to build {n} disjoint per-dimension pairs (got {pairs.shape[0]})")

    return pairs


def generate_mixed_pairs(total_pairs=15, dim=4, seed=0, global_ratio=0.4, per_dim_ratio=0.267, random_ratio=None):
    """
    Generate a set of pairs using a mixed strategy:
        - `global_ratio` (default 40%): max Euclidean distance across all dimensions
        - `per_dim_ratio` (default 26.7%): max distance per dimension (each dim has high variance)
        - `random_ratio` (default remaining): completely random pairs
    
    All points come from a common LHS pool (space-filling).
    
    For total_pairs=15 with defaults:
        - 6 global pairs (40%)
        - 4 per-dimension pairs (26.7%)
        - 5 random pairs (33.3%)
    """
    rng = np.random.default_rng(seed)

    # Compute counts from ratios
    if random_ratio is None:
        random_ratio = 1.0 - global_ratio - per_dim_ratio
    
    n_global = 6 # 6
    n_random = 4    # 4
    n_per_dim = 4   # 4
    
    if n_global < 0 or n_per_dim < 0 or n_random < 0:
        raise ValueError(f"Invalid pair counts: global={n_global}, per_dim={n_per_dim}, random={n_random}")

    # ----------------------------
    # 1) Generate a pool of points
    # ----------------------------
    sampler = qmc.LatinHypercube(d=dim, seed=rng)
    pool = sampler.random(n=2 * total_pairs)

    # -----------------------------------------
    # 2) Build sub-pools for each pairing method
    # -----------------------------------------
    # Each sub-pool must have exactly 2*N_i points
    idx_start = 0
    pool_global = pool[idx_start : idx_start + 2 * n_global]
    idx_start += 2 * n_global
    
    pool_per_dim = pool[idx_start : idx_start + 2 * n_per_dim]
    idx_start += 2 * n_per_dim
    
    pool_random = pool[idx_start : idx_start + 2 * n_random]
    
    # print(f"Generating {n_global} global (max Euclidean), {n_per_dim} per-dimension, and {n_random} random pairs.")

    # Build pairs
    pairs_global = (
        max_distance_pairs_from_pool(pool_global) if n_global > 0 else np.empty((0, 2, dim))
    )
    pairs_per_dim = (
        max_distance_per_dimension_pairs_from_pool(pool_per_dim, dim=dim, rng=rng) if n_per_dim > 0 else np.empty((0, 2, dim))
    )
    pairs_random = (
        make_pairs_from_pool(pool_random) if n_random > 0 else np.empty((0, 2, dim))
    )
    # print(pairs_per_dim)
    # ----------------------------
    # 3) Concatenate final pairs
    # ----------------------------
    pairs = np.concatenate([pairs_global, pairs_per_dim, pairs_random], axis=0)

    # Shuffle pairs so types are mixed
    # rng.shuffle(pairs)

    return torch.tensor(pairs, dtype=torch.float64)

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

pairs_to_test = generate_mixed_pairs(total_pairs=14, dim=4, seed=123)

ref = [58, 0.5, -46, 0.43]
ref_normalized = inverse_translate_param(ref)
# print("Reference point (normalized):", ref_normalized)
inv_ref = 1.0 - ref_normalized
inv_ref = np.clip(inv_ref, 0.0, 1.0)
# print("Inverted reference point (normalized):", inv_ref)
new_pair = np.vstack([ref_normalized, inv_ref])               # shape (2, dim)
new_pair_t = torch.tensor(new_pair, dtype=torch.float64).unsqueeze(0)  # shape (1,2,dim)
pairs_to_test = torch.cat([pairs_to_test, new_pair_t], dim=0)
# print("Generated initial pairs to test:", translate_param(pairs_to_test.numpy().reshape(-1, dim)))


""" OPTMIZATION LOOP PARAMETERS"""
n_queries = 40             # number of optimisation iterations
strategy = "EUBO"           # acquisition function
cofeedback_pt = None        # variable to store any cofeedback configuration
boolean_cofeedback = False   # can choose to activate cofeedback or not 

# for i, (a, b) in enumerate(pairs, 1):
#     print(f"\n===== PAIR {i} =====")
#     # print("Option 1:", a)
#     # print("Option 2:", b)

#     translated_p1 = translate_param(a)
#     translated_p2 = translate_param(b)

#     print("Option 1:", translated_p1)
#     print("Option 2:", translated_p2)

# n_init = 20                      # number of initial generated pair 

# rng1 = np.random.default_rng(seed=1)
# lhs_gen = qmc.LatinHypercube(d=dim, seed=rng1)
# points_lhs_1 = lhs_gen.random(n=n_init)

# rng2 = np.random.default_rng(seed=2)
# lhs_gen = qmc.LatinHypercube(d=dim, seed=rng2)
# points_lhs_2 = lhs_gen.random(n=n_init)
# np.random.shuffle(points_lhs_2)

# pairs_to_test = torch.tensor(
#     np.stack([points_lhs_1, points_lhs_2], axis=1),
#     dtype=torch.float64
# )



# for i in range(n_init):
#     p1 = pairs_to_test[i, 0].numpy()
#     p2 = pairs_to_test[i, 1].numpy()

#     translated_p1 = translate_param(p1)
#     translated_p2 = translate_param(p2)

#     print(f"\n===== PAIR {i+1} =====")
#     # print("Option 1:", np.round(translated_p1, 1).tolist())
#     # print("Option 2:", np.round(translated_p2, 1).tolist())
#     print("Option 1:", np.round(p1, 1).tolist())
#     print("Option 2:", np.round(p2, 1).tolist())

