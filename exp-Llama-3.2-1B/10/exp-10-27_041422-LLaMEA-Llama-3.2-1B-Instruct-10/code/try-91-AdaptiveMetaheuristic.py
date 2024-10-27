import numpy as np
import random

class AdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.current_sample = None
        self.current_sample_dir = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")
        
        if self.best_func is not None:
            return self.best_func
        
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 1
        
        # Set the sample size and directory
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Generate a random sample of size self.sample_size
            self.current_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the function at the current sample
            func_eval = func(self.current_sample)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 1:
                self.best_func = func_eval
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if func_eval > self.best_func:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            self.current_sample_dir = f"sample_{self.sample_size}_{_}"
            np.save(self.current_sample_dir, self.current_sample)
        
        return self.best_func

# **Selected Solution: Hyperband**
# Description: A metaheuristic algorithm that uses adaptive sampling to optimize black box functions using hyperband.
# Code: 
# ```python
# Hyperband
# ```
# ```python
# ```python
# # Description: A novel metaheuristic algorithm that uses adaptive sampling to optimize black box functions using hyperband.
# # Code: 
# # ```python
# def __init__(self, budget, dim):
#     self.budget = budget
#     self.dim = dim
#     self.best_func = None
#     self.best_func_evals = 0
#     self.sample_size = 1
#     self.sample_dir = None
#     self.current_sample = None
#     self.current_sample_dir = None
#     self.sample_history = []
#     self.iterations = 0

#     # Initialize the best function and its evaluation count
#     self.best_func = None
#     self.best_func_evals = 0

#     # Set the sample size and directory
#     self.sample_size = 10
#     self.sample_dir = f"sample_{self.sample_size}"
#     self.current_sample_dir = f"sample_{self.sample_size}_0"

#     # Perform adaptive sampling
#     for _ in range(self.budget):
#         # Generate a random sample of size self.sample_size
#         self.current_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
#         self.sample_history.append(self.current_sample)
        
#         # Evaluate the function at the current sample
#         func_eval = self.evaluate_func(self.current_sample)
        
#         # If this is the first evaluation, update the best function
#         if self.best_func_evals == 1:
#             self.best_func = func_eval
#             self.best_func_evals = 1
#         # Otherwise, update the best function if the current evaluation is better
#         else:
#             if func_eval > self.best_func:
#                 self.best_func = func_eval
#                 self.best_func_evals = 1
#             else:
#                 self.best_func_evals += 1

#     # Return the best function and its evaluation count
#     return self.best_func, self.best_func_evals

# def evaluate_func(self, func):
#     # Evaluate the function at the given sample
#     return func(self.current_sample)

# def update_sample(self, func):
#     # Update the current sample and save it to the sample directory
#     self.current_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
#     np.save(self.current_sample_dir, self.current_sample)

# def select_next_sample(self):
#     # Select the next sample from the sample history
#     if len(self.sample_history) > 0:
        # Get the last sample from the sample history
        # self.current_sample = self.sample_history[-1]
        # return self.current_sample
    #     else:
        #     # If the sample history is empty, generate a new sample
        #     self.current_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
        #     np.save(self.current_sample_dir, self.current_sample)
        #     return self.current_sample

# def select_next_best_func(self):
#     # Select the next best function from the sample history
#     if len(self.sample_history) > 0:
        # Get the last sample from the sample history
        # self.current_sample = self.sample_history[-1]
        # return self.best_func
    #     else:
        #     # If the sample history is empty, generate a new sample
        #     self.current_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
        #     np.save(self.current_sample_dir, self.current_sample)
        #     return self.best_func

# def select_next_best_func_evals(self):
#     # Select the next best function and its evaluation count from the sample history
#     if len(self.sample_history) > 0:
        # Get the last sample from the sample history
        # self.current_sample = self.sample_history[-1]
        # return self.best_func_evals, self.best_func_evals
    #     else:
        #     # If the sample history is empty, generate a new sample
        #     self.current_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
        #     np.save(self.current_sample_dir, self.current_sample)
        #     return 1, 1

# def __call__(self, func):
#     # Evaluate the function at the given sample
#     return self.select_next_best_func()

# def update(self, func, budget, dim):
#     # Update the sample and the best function
#     self.update_sample(func)
#     self.best_func, self.best_func_evals = self.select_next_best_func()

# def run(self, func, budget, dim):
#     # Run the optimization algorithm
#     return self.__call__(func), self.update(budget, dim)

# **Code:**