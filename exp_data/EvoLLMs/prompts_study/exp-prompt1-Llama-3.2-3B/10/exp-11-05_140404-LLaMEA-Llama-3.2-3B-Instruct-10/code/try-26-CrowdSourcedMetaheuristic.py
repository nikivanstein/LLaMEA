import numpy as np
from scipy.optimize import differential_evolution

class CrowdSourcedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.array([-5.0, 5.0])  # Search space between -5.0 and 5.0
        self.mean = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)  # Initialize mean
        self.covariance = np.eye(self.dim) * 1.0  # Initialize covariance matrix
        self.hyperparameters = {
            'evolution_strategy_std': 1.0,
            'genetic_drift_covariance': 0.1,
            'hybrid_evolution_ratio': 0.2
        }

    def __call__(self, func):
        for _ in range(self.budget):
            # Perform evolution strategy to update mean
            new_mean = self.mean + np.random.normal(0, self.hyperparameters['evolution_strategy_std'], size=self.dim)
            new_mean = np.clip(new_mean, self.search_space[0], self.search_space[1])  # Clip values to search space

            # Perform genetic drift to update covariance
            new_covariance = self.covariance + np.random.normal(0, self.hyperparameters['genetic_drift_covariance'], size=(self.dim, self.dim))
            new_covariance = np.clip(new_covariance, 0, 1.0)  # Clip values to avoid negative covariance matrix

            # Evaluate function at new mean
            f_new = func(new_mean)

            # Update mean and covariance
            self.mean = new_mean
            self.covariance = new_covariance

            # Print current best solution
            print(f"Current best solution: x = {self.mean}, f(x) = {f_new}")

        # Hybrid differential evolution with evolution strategy and genetic drift
        bounds = [(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=self.mean, seed=42, tol=1e-6)
        print(f"Final best solution: x = {res.x}, f(x) = {res.fun}")

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

crowd_sourced = CrowdSourcedMetaheuristic(budget=100, dim=2)
crowd_sourced(func)

# Refinement:
# - Added hyperparameters to control evolution strategy and genetic drift
# - Hybridized differential evolution with evolution strategy and genetic drift