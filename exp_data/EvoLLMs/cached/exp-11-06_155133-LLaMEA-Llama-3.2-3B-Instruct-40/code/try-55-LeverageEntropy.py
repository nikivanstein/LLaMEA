import numpy as np
import random

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

    def __call__(self, func):
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0

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

        # Reduce the entropy to maintain the balance between exploration and exploitation
        self.entropy = max(0.0, self.entropy - 0.1)

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        # Select the dimension with the highest leverage
        leverage = np.array([entropy / (x[i]!= self.lower_bound and x[i]!= self.upper_bound) for i in range(self.dim)])
        self.dim_leverage = np.argmax(leverage)
        self.x_best = np.insert(self.x_best, self.dim_leverage, self.x_best[self.dim_leverage])

        return self.f_best

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    print(leverage_entropy(func))

# Refine the strategy of the selected solution
# Refine the algorithm by introducing a new dimension to leverage the best solution
# ```python
class LeverageEntropy:
    #...

    def __call__(self, func):
        #...

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        # Select the dimension with the highest leverage
        leverage = np.array([entropy / (x[i]!= self.lower_bound and x[i]!= self.upper_bound) for i in range(self.dim)])
        self.dim_leverage = np.argmax(leverage)
        self.x_best = np.insert(self.x_best, self.dim_leverage, self.x_best[self.dim_leverage])

        # Leverage the best solution
        best_x = np.array([self.x_best[i] if i!= self.dim_leverage else self.lower_bound for i in range(self.dim)])
        best_f = func(best_x)
        self.entropy -= np.sum([1 / np.log(2 * np.pi * np.sqrt(1 + (best_x[i] - self.lower_bound) ** 2)) for i in range(self.dim) if best_x[i]!= self.lower_bound and best_x[i]!= self.upper_bound])
        self.f_best = min(self.f_best, best_f)

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    print(leverage_entropy(func))