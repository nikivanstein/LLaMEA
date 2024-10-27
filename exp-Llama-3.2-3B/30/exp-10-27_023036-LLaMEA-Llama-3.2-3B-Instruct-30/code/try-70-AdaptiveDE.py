import numpy as np
import random

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.x_best = np.zeros(self.dim)
        self.f_best = np.inf

    def __call__(self, func):
        if self.budget == 0:
            return self.x_best

        self.x = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.f = np.zeros(self.population_size)

        for i in range(self.population_size):
            self.f[i] = func(self.x[i])

        for _ in range(self.budget):
            # Select the best individual
            idx = np.argmin(self.f)
            self.x_best = self.x[idx]
            self.f_best = self.f[idx]

            # Create a new population
            new_x = np.zeros((self.population_size, self.dim))
            new_f = np.zeros(self.population_size)

            # Adaptive Differential Evolution
            for j in range(self.population_size):
                if random.random() < 0.3:
                    # Refine the mutation strategy
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    new_x[j] = self.x[j] + r1 * (self.x[idx] - self.x[j]) + r2 * (self.x[(idx + 1) % self.population_size] - self.x[j])
                else:
                    new_x[j] = self.x[j]

                new_f[j] = func(new_x[j])

            # Update the population
            self.x = new_x
            self.f = new_f

            # Update the best individual
            self.f_best = np.min(self.f)

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
de = AdaptiveDE(budget, dim)
x_best = de(func)
print(x_best)