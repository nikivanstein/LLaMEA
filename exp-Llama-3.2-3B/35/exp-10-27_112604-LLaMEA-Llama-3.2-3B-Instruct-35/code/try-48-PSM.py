import numpy as np
import random

class PSM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.probability = 0.35

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the function at the current particles
            values = func(self.particles[i])
            # Update the personal best and global best
            self.pbest[i] = self.particles[i]
            self.rbest[i] = values
            # Check if the current best is better than the stored best
            if np.any(values < self.rbest, axis=1):
                self.memory[i] = values
                self.rbest[np.argmin(self.rbest, axis=1)] = values
            # Update the particles using the PSO update rule
            self.particles[i] = self.particles[i] + 0.5 * (self.pbest[i] - self.particles[i]) + 0.5 * (self.rbest - self.particles[i])
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

    def refine_strategy(self):
        for i in range(self.budget):
            if random.random() < self.probability:
                # Randomly select a particle
                j = random.randint(0, self.budget - 1)
                # Randomly select a dimension
                k = random.randint(0, self.dim - 1)
                # Randomly select a value between -5.0 and 5.0
                new_value = random.uniform(-5.0, 5.0)
                # Update the particle
                self.particles[i, k] = new_value
                # Update the personal best
                self.pbest[i, k] = self.particles[i, k]
                # Update the global best
                self.rbest[i, k] = self.particles[i, k]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)