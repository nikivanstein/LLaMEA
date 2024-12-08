import numpy as np
import random
import os

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.sample_dir_evals = 0

    def __call__(self, func, logger):
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
        self.sample_dir_evals = 0

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
            self.save_sample(sample)

            # Increment the evaluation count
            self.sample_dir_evals += 1

            # If the evaluation count reaches the budget, perform evaluation
            if self.sample_dir_evals == self.budget:
                self.evaluate_func(func_eval, logger)

        return self.best_func

    def save_sample(self, sample):
        np.save(f"{self.sample_dir}_{self.sample_dir_evals}_{random.randint(0, 1000)}", sample)

    def evaluate_func(self, func_eval, logger):
        # Evaluate the function using the sample
        func_eval = np.linalg.norm(func_eval - np.mean(func_eval))  # Use mean of function evaluations
        logger.info(f"Function evaluation: {func_eval:.4f}")

    def sample(self, func, logger):
        # Perform adaptive sampling
        while True:
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

            # Increment the evaluation count
            self.sample_dir_evals += 1

            # If the evaluation count reaches the budget, perform evaluation
            if self.sample_dir_evals == self.budget:
                self.evaluate_func(func_eval, logger)

            # Break the loop if the function value converges
            if np.abs(func_eval - self.best_func) < 1e-6:
                break

        # Update the best function
        self.best_func = func_eval
        self.best_func_evals = 1