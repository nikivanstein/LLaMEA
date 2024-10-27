import numpy as np
import random

class PSEM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.evolution_rate = 0.35

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
            # Apply evolutionary adaptation
            if random.random() < self.evolution_rate:
                # Select a random particle
                j = random.randint(0, self.budget - 1)
                # Replace the particle with a new one
                self.particles[j] = np.random.uniform(-5.0, 5.0, (1, self.dim))
                # Update the personal best and global best
                self.pbest[j] = self.particles[j]
                self.rbest[j] = func(self.particles[j])
                # Check if the new particle is better than the stored best
                if np.any(self.rbest[j] < self.rbest, axis=1):
                    self.memory[j] = self.rbest[j]
                    self.rbest[np.argmin(self.rbest, axis=1)] = self.rbest[j]
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)