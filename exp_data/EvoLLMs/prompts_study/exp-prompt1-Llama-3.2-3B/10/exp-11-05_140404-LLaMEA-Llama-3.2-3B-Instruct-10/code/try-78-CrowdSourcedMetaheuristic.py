import numpy as np
from scipy.optimize import differential_evolution
import random

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.covariance_update_rate = int(self.budget * 0.5)  # Update covariance at 50% of the budget
        self.population_size = 10  # Increase population size for better exploration
        self.adaptive_covariance = True  # Use adaptive covariance update
        self.mutation_rate = 0.1  # Mutation rate for evolution strategy
        self.crossover_rate = 0.5  # Crossover rate for genetic drift

    def __call__(self, func):
        for i in range(int(self.budget * self.covariance_update_rate)):
            # Perform evolution strategy to update mean
            if np.random.rand() < self.mutation_rate:
                new_mean = self.mean + np.random.normal(0, 1.0, size=self.dim)
                new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space
                new_mean = self.sample_strategy(new_mean)  # Apply sampling strategy
            else:
                new_mean = self.mean

            # Perform genetic drift to update covariance
            if np.random.rand() < self.crossover_rate:
                new_covariance = self.covariance + np.random.normal(0, 0.1, size=(self.dim, self.dim))
                new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix
                new_covariance = self.update_covariance(new_covariance)
            else:
                new_covariance = self.covariance

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Print current best solution
            print(f"Current best solution: x = {self.mean}, f(x) = {f_new}")

        # Perform final optimization using differential evolution
        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=self.mean, seed=42)
        print(f"Final best solution: x = {res.x}, f(x) = {res.fun}")

    def sample_strategy(self, mean):
        # Apply adaptive sampling strategy
        if self.adaptive_covariance:
            # Calculate the variance of the mean
            variance = np.var(mean)
            # Sample from a normal distribution with mean and variance
            new_mean = mean + np.random.normal(0, variance, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space
        else:
            # Randomly sample from the mean
            new_mean = mean + np.random.normal(0, 1.0, size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space
        return new_mean

    def update_covariance(self, covariance):
        # Update covariance matrix
        new_covariance = covariance + 0.01 * np.eye(self.dim)  # Add a small value to covariance matrix
        new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix
        return new_covariance

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(func)
