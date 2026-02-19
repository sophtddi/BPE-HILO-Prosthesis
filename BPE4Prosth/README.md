# Preference-Based Bayesian Optimization Framework

This project implements a preference-driven Bayesian Optimization (BO) workflow designed to learn user preferences through pairwise comparisons. The system iteratively proposes configuration pairs, asks the user which one they prefer, and updates a Gaussian Process model built on pairwise data.

This framework is designed to learn user preferences efficiently using minimal interactions. It is particularly suited for situations where:
- Users can't provide numeric scores
- Only pairwise judgments ("I prefer A over B") are available
- Parameters must be tuned to a user’s perception or comfort (e.g., prosthesis settings)

# Algorithm presentation
It is structured into two main phases:

- Initial Design (ID) – collecting n_init random initial comparison data based on LHS 
- Exploration – iteratively proposing new, informative pairs using an acquisition function (EUBO)

1) Probabilistic model of the user preference

The model is a Gaussian process (GP) for preference learning using pairwise comparisons based on a GP prior distribution and a likelihood function.

The prior is based on a Matern kernel (nu=5/2) and a zero mean function (ConstantMean()). 

The likelihood is modeled by a PairwiseLogitLikelihood.

Due to the combination of the GP prior and likelihood, the posterior is not analytically tractable. Instead, it is approximated using Laplace approximation.

2) User interaction

At each iteration, the algorithm select three points that optimize the EUBO acquisition function onstructed from the current posterior distribution. The first two candidates are proposed in a random order to the user. Those data are then used to update the model.
If the user has no preference, at next iteration, the algorithm presents the first candidate against the third candidate of previous iterations. If again no preference is provided, the user is asked his preference between the first candidate and a random point, until they can express a preference.

3) Convergence criteria 

The algorithm is considered to have converged when the maximum utility point (estimate of user preference) stay in the same 10% region in all dimension for n_stops iterations.

4) Data structure

At each iteration, the user is requested his preference between two configurations [previous_preference, challenger]. Those visited points are stored in a list (train_x) and the preference is stored in (train_comp). In train_x, every configuration is only stored once, eventhough the configuration is tested multiple time. In train_comp, the preference is stored as [i,j] with i and j the indices of previous_preference and challenger in train_comp. So for example, if train_x = [x1, x2] and the query is "x1 or x2?", train_comp would be [0,1] if x1 is prefered to x2. At next iteration, when testing a challenger x3 against the prefered configuration x1, the query is now "x1 or x3?" and train_comp should be updated by concanating it with [2,0]. Therefore, train_x would then be [x1, x2, x3] and train_comp [[0,1], [2,0]].

# PROJECT STRUCTURE
The project is organized into several modules that separate configuration, core utilities, initial design logic, and exploration logic:

*config.py*: Contains global configuration variables, parameter ranges,... 

*core_function.py*: Includes core utility functions shared across the entire pipeline. 

*ID_functions.py*: Contains helper functions for the Initial Design (ID) phase. 

*ID.py*: Implements the full Initial Design pipeline. 

*exploration_function.py*: Contains functions used during the Exploration phase 

*exploration.py*: Implements the full Exploration loop.

This project is built on top of BoTorch, leveraging its tools for preference-based Bayesian Optimization, Gaussian Process modeling, and acquisition strategies. All required dependencies—including BoTorch, PyTorch, and supporting libraries—are specified in the pyproject.toml file for reproducibility and easy environment setup.

**Important note:** the default PairwiseLaplaceMarginalLogLikelihood class from BoTorch has been temporarily modified in this project to incorporate a noise likelihood term. This customization is necessary for the current experimental setup and should be kept in mind when updating or reinstalling BoTorch.
