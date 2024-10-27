import numpy as np
from scipy.optimize import differential_evolution
from typing import Dict

class AdaptiveHyperband:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"

    def __call__(self, func: callable) -> callable:
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
            sample = np.random.uniform(-5.0, 5.0, size=self.dim)

            # Evaluate the function at the current sample
            func_eval = func(sample)

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
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)

        # Define an evolutionary strategy to refine the best function
        def evolutionary_strategy(individual: np.ndarray) -> np.ndarray:
            # Sample from the current population
            sample = np.random.choice(self.best_func_evals, size=self.sample_size, replace=False)

            # Perform differential evolution to refine the best function
            result = differential_evolution(lambda x: np.mean(x), [(x - 1) / 10 for x in sample], x0=individual)

            return result.x

        # Update the best function using the evolutionary strategy
        self.best_func = differential_evolution(lambda x: np.mean(x), [(x - 1) / 10 for x in evolutionary_strategy(individual)], x0=individual)[0]

        return self.best_func