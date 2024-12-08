import numpy as np
import random

class GBestPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 2.049912
        self.f = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fval = np.inf
        self.best_x = np.inf
        self.selection_prob = 0.35

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]
                self.x = np.array([self.best_x])

            # Select the best individual
            selected_indices = np.random.choice(self.population_size, size=int(self.population_size * self.selection_prob), replace=False)
            self.x = self.x[selected_indices]

            # Apply PSO and DE operators
            v = self.w * np.random.uniform(0, 1, (self.population_size, self.dim)) + self.c1 * np.abs(self.x - self.best_x[:, np.newaxis]) + self.c2 * np.abs(self.x - np.mean(self.x, axis=0)[:, np.newaxis]) ** self.f
            self.x = self.x + v

            # Limit the search space
            self.x = np.clip(self.x, self.lower_bound, self.upper_bound)

            # Evaluate the function at the updated population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]

        return self.fval, self.best_x

# Example usage:
def func(x):
    return np.sum(x**2)

gbest_pso_de = GBestPSODE(budget=100, dim=10)
fval, best_x = gbest_pso_de(func)
print(f"Best function value: {fval}, Best solution: {best_x}")