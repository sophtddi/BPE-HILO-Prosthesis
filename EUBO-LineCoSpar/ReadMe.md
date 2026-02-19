# User-Guided Pairwise Bayesian Optimization

This project implements a **user-in-the-loop optimization framework** for tuning control parameters of a prosthesis using **Pairwise Bayesian Optimization (BO)**. The approach combines an initial **exploration phase** to build a surrogate model of user preference, followed by an **optimization phase** to iteratively converge to the user's preferred configuration and a **validation phase** to check user preference consistency. The model is built on **pairwise preferences** rather than numeric scores, using a **Pairwise Gaussian Process (PairwiseGP)** trained with a **Laplace approximation**.

The model is trained on a n_dim normalized grid containing n_grid points per dimension, which makes n_grid**n_dim possible normalized configurations. The normalized configurations can be translated back to the real range of values of the parameters by using the `translate_param` function depending on the case of application (defined in the `config.py` file).

	1) Probabilistic model of the user preference

The model is a Gaussian process (GP) for preference learning using pairwise comparisons based on a GP prior distribution and a likelihood function.

The prior is based on a RBF kernel and a zero mean function (ConstantMean()). The kernel parameters such as lengthscale and outputscale are estimated at evry iterations by maximising the log-marginal likelihood.

	- The lenghtscale is initialized as the distance between two closest point in the grid and is then fitted at each iteration based on a LogNormal prior centered around the lengthscale at previous iteration. Moreover, the lengthscale is bounded between the longest distance between two points of the grid, and  the smallest distance between two points of the grid.
	- The outputscale is first fitted based on the initial exploration queries, and is then fitted at each iteration based on a LogNormal prior centered around the outputscale at previous iteration. Moreover, the outputscale is arbitrarly bounded between 1 and 100.

The likelihood is modeled by a PairwiseProbitLikelihood. 

Due to the combination of the GP prior and likelihood, the posterior is not analytically tractable. Instead, it is approximated using Laplace approximation.

	2) User interaction
The user start to select a preferred configuration among an initial random pair. At each iteration, the algorithm has to select a challenger to this prefered configuration. The challenger to be presented to the user is chosen by maximizing an acquisition function constructed from the current posterior distribution. In this case, EUBO is used as the acquisition function. The user can then give his preference between his previous prefered configuration and the challenger. Those data are then used to update the model.

In practice, to increase the exploration/exploitation trade-off, the acquisition function is only evaluated on some specific points. The points considered are as follows: either they are points already visited during previous iterations, or they are points located in the neighborhood of a random direction line passing through the current preferred point. The neighborhood of the line is defined by the 'radius' parameter, which is set to 1.1 by default to consider points at a distance of one step of the maximum line discretization, but the parameter can be increased to take into account more configurations as well.

Moreover, after every comparison, the user can also give a cofeedback by orienting the next challenger. 
A cofeedback takes the following form: the user can suggest an increment or decrement of a discretization step in one of the dimensions from the point they have just defined as their preference.
If a cofeedback is given, then the next challenger is the configuration suggested by the user, and the acquisition function is not used.

	3) Convergence criteria 
The algorithm is considered to have converged when the same point is kept as the preference for n_stop iterations. To force a certain amount of exploration, this criterion only begins to be evaluated after n_min iterations. 

After convergence, the user preference is tested in a validation phase. The user is presented with n_verif pairs, each consisting of the convergence preferred point and a random challenger, presented in random order.

	4) Data structure
At each iteration, the user is requested his preference between two configurations [previous_preference, challenger]. Those visited points are stored in a list (train_x) and the preference is stored in (train_comp). In train_x, every configuration is only stored once, eventhough the configuration is tested multiple time. In train_comp, the preference is stored as [i,j] with i and j the indices of previous_preference and challenger in train_comp. So for example, if train_x = [x1, x2] and the query is "x1 or x2?", train_comp would be [0,1] if x1 is prefered to x2. At next iteration, when testing a challenger x3 against the prefered configuration x1, the query is now "x1 or x3?" and train_comp should be updated by concanating it with [2,0]. Therefore, train_x would then be [x1, x2, x3] and train_comp [[0,1], [2,0]].

# MODEL SUMMARY
**Model Type**: Pairwise Gaussian Process (`PairwiseGP`)

**Likelihood**: Based on pairwise comparisons using `PairwiseLikelihood`

**Kernel**: RBF kernel with learned `lengthscale` and `outputscale`

**Inference**: Laplace Approximation to obtain posterior over latent utility function

**Acquisition Function**: Expected Utility of the Best Option (EUBO)

The user can also guide the model exploration by providing a cofeeback if boolean_cofeedback=True. The feedback can consist of an increase or a decrease in one selected parameters (for example, an increase would correspond to the saame configuration with only one parameter modified to the higher discretized value). 

# WORKFLOW SUMMARY 
Step 1: Run exploration to collect initial user preferences ->  python exploration.py

Step 2: Run optimization -> python optimization.py

Step 3: Run validation -> python optimization.py


# PROJECT STRUCTURE 
The codebase consists of 7 files, each responsible for a distinct part of the pipeline:

 --- `config.py` ---
 
Defines global parameters and configuration setups used throughout exploration, optimization, and validation.  
Key responsibilities: 
- Generate and store the a normalized grid corresponding to all configurations with `generate_grid_points` (by default dim=4, n_grid=5)
- Define the real range of values of each parameters  
- Configure default values for model training, such as kernel constraints
- Define the n_init initial pairs for the exploration phase
- Define optimization loop hyperparameters (n_queries, n_stop, boolean to enable cofeedback, n_verif,...)


 --- `exploration.py` ---
 
Defines the **pipeline for the exploration phase**, which builds the initial Gaussian Process (GP) model using user preferences over selected pairs defined in `config.py`.


 --- `optimization.py` ---
 
Defines the **main pipeline for the optimization and validation trial**.
1. **Model Initialization**:
   - Loads the model trained during exploration.
   - Begins with a random pair of configurations and asks for user preference.
2. **Iterative Optimization**:
   For max n_queries iterations:
   - Builds the posterior using the updated GP.
   - If no user feedback at previous iteration, selects a challenger by computing EUBO over points:
     - That lie on a line through the current best in a random direction.
     - And/or were already visited.
   - Compares the challenger against the current best based on user preference.
   - Updates the model and continues until convergence.
   - Ask user for a potential feedback for next iteration
   Stopping criteria: after n_min iterations, when a same configuration is kept as preference for n_stop iterations  
3. **Validation Phase**:
   - After convergence, the final configuration is validated by comparing it to n_verifs randomly selected alternative.


 --- `core_functions.py` ---
 
Contains core utility functions shared by both the exploration and optimization processes. Main functions include:
- `compare_objective_values(pair)`: Asks the user to select the preferred configuration in a pair.
- `user_preference(pair)`: Extracts user preference and updates the best configuration accordingly.
- `update_model(model, mll, train_x, train_comp)`: Optimizes the model by maximizing the marginal log-likelihood using Laplace approximation.
- `append_data(x_next, x_train, comp_next, comp_train)`: Adds new comparison data, handling duplicates in the training set.
- `translate_param(x)`: Converts normalized input vectors into their real-world parameter equivalents.
- `save_model(...)`: Saves the current model state and data to disk for reproducibility or later analysis.


 --- `exploration_functions.py` ---
 
Handles the **exploration phase**, where the user is asked to compare predefined parameter pairs to construct an initial surrogate model.
- `problem_exploration(xs, pairs_to_test, l, length_bounds, out_bounds)`: Queries user preferences over a set of parameter pairs and builds an initial GP model using those preferences.


 --- `optimization_function.py` ---
 
Contains specific helper functions for running the **optimization loop**.
- `initial_acquisition(xs)`: Randomly selects the first pair of configurations.
- `acquisition(xs, model, current_best, visited_pairs, max_attempts)`: Selects the next challenger by maximizing the Expected Utility of the Best Option (EUBO).
- `cofeedback(xs, current_best, ub, lb, n_grid, pending_coactive_pt)`: Incorporates fine-grained user feedback based on local parameter modifications.
- `early_stopping`: Stops optimization when no improvement is observed for a defined number of steps.
- `get_random_direction(dim)`: Generates a random unit direction vector in parameter space.
- `points_along_line(best_point, direction, grid_points, visited_points)`: Selects candidate configurations along a line in a randomly sampled direction around the current best, plus previously visited points.


 --- `validation_function.py` ---
 
This file contains specific functions used during the validation phase of the experiment in the optimization.py file to evaluate the consistency of user preference between his prefered point and a random challenger.
- `validation_test(best_translate, challenger, i)`:Evaluate the consistency of user preference between his prefered point and a random challenger.






