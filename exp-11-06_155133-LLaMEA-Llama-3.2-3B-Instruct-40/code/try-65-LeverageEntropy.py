import numpy as np
import random
from scipy.stats import norm

class LeverageEntropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.covariance_matrix = np.zeros((self.dim, self.dim))
        self.covariance_update_count = 0

    def __call__(self, func):
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.covariance_matrix = np.zeros((self.dim, self.dim))

        for _ in range(self.budget):
            # Randomly select a dimension to leverage
            dim = random.randint(0, self.dim - 1)

            # Generate a random point in the search space
            x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            # Calculate the entropy of the current point
            entropy = 0.0
            for i in range(self.dim):
                if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                    entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))

            # Update the entropy
            self.entropy += entropy

            # Evaluate the function at the current point
            f = func(x)

            # Update the best solution if the current solution is better
            if self.f_best is None or f < self.f_best:
                self.f_best = f
                self.x_best = x

            # If the current solution is close to the best solution, reduce the entropy
            if self.f_best_val - f < 1e-3:
                self.entropy -= entropy / 2

            # Update the covariance matrix
            self.covariance_matrix += np.outer(x - self.x_best, x - self.x_best)
            self.covariance_update_count += 1

        # Update the covariance matrix
        if self.covariance_update_count > 0:
            self.covariance_matrix /= self.covariance_update_count

        # Compute the covariance matrix inverse
        try:
            self.cov_matrix_inv = np.linalg.inv(self.covariance_matrix)
        except np.linalg.LinAlgError:
            # If the covariance matrix is singular, use a small value as the inverse
            self.cov_matrix_inv = np.eye(self.dim) / self.dim

        # Compute the exploration term
        exploration_term = np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix_inv)

        # Compute the exploitation term
        exploitation_term = self.x_best + np.random.uniform(-0.1, 0.1, self.dim)

        # Combine the exploration and exploitation terms
        x_new = exploration_term * np.sqrt(0.1) + exploitation_term

        # Evaluate the function at the new point
        f_new = func(x_new)

        # Update the best solution if the new solution is better
        if self.f_best is None or f_new < self.f_best:
            self.f_best = f_new
            self.x_best = x_new

        # Reduce the entropy to maintain the balance between exploration and exploitation
        self.entropy = max(0.0, self.entropy - 0.1)

        return self.f_best

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    print(leverage_entropy(func))
