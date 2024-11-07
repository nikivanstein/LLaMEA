import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import minimize

class ImprovedLeverageEntropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.entropy_history = []
        self.exploitation_rate = 0.2
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        self.alpha = 0.1
        self.bayesian_optimizer = BayesianOptimizer(self.budget, self.dim)
        self.lr = 0.05  # adaptive learning rate

    def __call__(self, func):
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.entropy_history = []

        for _ in range(self.budget):
            # Generate a random point in the search space
            x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            # Calculate the entropy of the current point
            entropy = 0.0
            for i in range(self.dim):
                if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                    entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))

            # Update the entropy
            self.entropy += entropy
            self.entropy_history.append(self.entropy)

            # Evaluate the function at the current point
            f = func(x)

            # Update the best solution if the current solution is better
            if self.f_best is None or f < self.f_best:
                self.f_best = f
                self.x_best = x

            # If the current solution is close to the best solution, reduce the entropy
            if self.f_best_val - f < 1e-3:
                self.entropy -= entropy / 2

            # Update the mean and covariance of the Bayesian optimization
            self.mean = self.alpha * self.mean + (1 - self.alpha) * x
            self.cov = self.alpha * np.eye(self.dim) + (1 - self.alpha) * np.eye(self.dim) - (1 - self.alpha) * np.outer(x - self.mean, x - self.mean) / (self.dim * (self.dim + 1))

            # Sample a new point using the Bayesian optimization
            x_new = self.bayesian_optimizer.sample()
            f_new = func(x_new)

            # Update the best solution if the new solution is better
            if self.f_best_val > f_new:
                self.f_best = f_new
                self.x_best = x_new

            # Update the learning rate
            self.lr = max(0.01, self.lr * 0.9)  # decay the learning rate

        # Reduce the entropy to maintain the balance between exploration and exploitation
        self.entropy = max(0.0, self.entropy - 0.1)

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        return self.f_best

class BayesianOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.entropy_history = []
        self.exploitation_rate = 0.2
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim)
        self.alpha = 0.1
        self.max_iter = 10
        self.min_iter = 1
        self.lr = 0.05  # adaptive learning rate

    def sample(self):
        # Perform Bayesian optimization
        def neg_func(x):
            f = func(x)
            return -f

        res = minimize(neg_func, self.mean, method="L-BFGS-B", bounds=[(self.lower_bound, self.upper_bound)] * self.dim, tol=self.lr)
        x_new = res.x

        # Update the mean and covariance
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x_new
        self.cov = self.alpha * np.eye(self.dim) + (1 - self.alpha) * np.eye(self.dim) - (1 - self.alpha) * np.outer(x_new - self.mean, x_new - self.mean) / (self.dim * (self.dim + 1))

        # Sample a new point
        epsilon = np.random.normal(0, 1)
        x_new = self.mean + np.sqrt(np.diag(self.cov)) * np.random.normal(0, 1, self.dim) + epsilon * np.diag(self.cov) ** 0.5

        return x_new

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
improved_leverage_entropy = ImprovedLeverageEntropy(budget, dim)
for _ in range(100):
    print(improved_leverage_entropy(func))
