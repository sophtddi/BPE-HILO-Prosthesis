# Preference-Based Prosthesis Control Tuning

This repository contains the code for two **preference-based Bayesian optimization algorithms** designed to personalize transfemoral prosthesis control using human-in-the-loop feedback with pairwise comparison. These algorithms were used to run real-world experiments with active prostheses and are implemented for real-life tuning.

## Algorithms

1. **EUBO-LineCoSpar**  
   - A discretized version of the algorithm that simplifies the parameter space.  
   - Based on the LineCoSpar framework ([Tucker et al., 2020](https://doi.org/…)), replacing Thompson Sampling with **Expected Utility of the Best Option (EUBO)** for improved sample efficiency.  
   - Suitable for faster prototyping and lower computational overhead.

2. **BPE4Prosth**  
   - Extends the framework to a continuous parameter space with a **Logit likelihood**, allowing richer exploration.  
   - Handles noisy feedback more robustly, improving convergence and personalization.  
   - Designed for real-world prosthesis control tuning with human subjects.

