import numpy as np
import random

class PSML:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.learning_rate = 0.35

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
            # Apply learning from experience
            for j in range(self.dim):
                if random.random() < self.learning_rate:
                    self.particles[i, j] += np.random.uniform(-0.1, 0.1)
            # Ensure the particles are within the search space
            self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]