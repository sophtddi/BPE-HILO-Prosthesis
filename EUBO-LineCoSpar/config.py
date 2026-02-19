"""
This file contains functions and parameters for generating grid points and defining the setup for exploration, optimization and validation processes. 
"""

import os
import torch

def generate_grid_points(dim, lb, ub, num_points_per_dim):
    """
    Generate the grid on which to perform optimisation
    """
    if isinstance(num_points_per_dim, int):
        num_points_per_dim = [num_points_per_dim] * dim
    
    # Ensure num_points_per_dim has the right length
    assert len(num_points_per_dim) == dim, "Length of num_points_per_dim must match dim"
    
    # Convert lb and ub to tensors if they're not already
    if not isinstance(lb, torch.Tensor):
        lb = torch.tensor([lb] * dim)
    if not isinstance(ub, torch.Tensor):
        ub = torch.tensor([ub] * dim)
    # Create a list of axes, one for each dimension
    axes = []
    for i in range(dim):
        axes.append(torch.linspace(lb[i], ub[i], num_points_per_dim[i]))
    
    # Generate grid points
    grid = torch.meshgrid(*axes, indexing='ij')
    points = torch.stack(grid, dim=-1).reshape(-1, dim)
    
    return points



""" SAVING PARAMETERS """
save_dir = 'test'
explo_file = 'explo_data.pt'
file_path = os.path.join(save_dir, explo_file)


""" GRID DEFINITION """
dim = 4             # dimension of the problem
lb = 0              # lower bound
ub = 1              # upper bound
n_grid = 5          # number of points in the grid per dimension
all_config = generate_grid_points(dim=dim, lb=lb, ub=ub, num_points_per_dim=n_grid)

params1 = torch.tensor([40, 45, 50, 55, 60], dtype=torch.float32)         # swing flexion timing t1 range of values
params2 = torch.tensor([0.4, 0.8, 1.1, 1.4, 1.8], dtype=torch.float32)    # swing flexion stiffness k1 range of values
params3 = torch.tensor([40, 45, 50, 55, 60], dtype=torch.float32)         # flexion equilibrium position theta1 range of values
params4 = torch.tensor([0.3, 0.5, 0.7, 0.9, 1.1], dtype=torch.float32)    # swing extension stiffness k2 range of values

""" MODEL HYPERPARAMETERS """
# RBF kernel k(x,x') = outpuscale² * exp(-||x-x'||²/(2*lengthscale²)
length_bounds = (0.1, 2.0)      # lenghtscale bounds of the kernel (lower bound should be lower than grid spacing and higher bound should be the grid max distance)
out_bounds = (1, 100.0)         # outpuscale bounds of the kernel
prior_scale = 0.5               # prior to optimize both lengthscale and outpuscale with fit_gpytorch_mll() 

"""" EXPLORATION PARAMETERS """
lengthscale = 0.25              # initial lengthscale of the model (here defined as grid spacing)
n_init = 5                      # number of initial generated pair 
pairs_to_test = [
    ([0.00, 0.50, 0.00, 0.25], [1.00, 0.50, 1.00, 0.25]),  # dim 1–3 interaction
    ([0.25, 0.50, 0.75, 0.25], [0.75, 0.50, 0.25, 0.25]),  # anti-diag 1–3
    ([0.50, 0.00, 0.50, 0.75], [0.50, 1.00, 0.50, 0.75]),  # dim 2
    ([0.50, 0.50, 0.50, 0.00], [0.50, 0.50, 0.50, 0.50]),  # dim 4
    ([0.00, 0.00, 0.00, 0.00], [1.00, 1.00, 1.00, 1.00])   # diag 4D
]


""" OPTMIZATION LOOP PARAMETERS"""
n_queries = 15              # number of optimisation iterations
n_stop = 3                  # number of iterations to define convergence (after n_stop iterations with the same best, stop)
n_min = 3                   # number of minimum iteration before checking convergence condition
strategy = "EUBO"           # acquisition function
cofeedback_pt = None        # variable to store any cofeedback configuration
boolean_cofeedback = True   # can choose to activate cofeedback or not 
radius = 1.1                # radius × grid spacing defines how far a point can be from the random direction line to be considered a candidate

""" VALIDATION PARAMETERS """
n_verif = 3                 # number of test in validation session





