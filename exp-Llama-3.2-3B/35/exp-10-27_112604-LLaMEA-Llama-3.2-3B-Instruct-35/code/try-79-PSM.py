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
        self.adaptation_rate = 0.35

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
            # Adapt the particles based on the best solution found
            if i > 0 and np.any(values < self.rbest[np.argmin(self.rbest, axis=1)]):
                # Select particles to adapt
                adapt_particles = random.sample(range(self.budget), int(self.budget * self.adaptation_rate))
                # Update the particles
                for j in adapt_particles:
                    # Select a random dimension to adapt
                    dim_to_adapt = random.randint(0, self.dim - 1)
                    # Generate a new value for the dimension
                    new_value = np.random.uniform(-5.0, 5.0)
                    # Update the particle
                    self.particles[j, dim_to_adapt] = new_value
                    # Update the personal best and global best
                    values = func(self.particles[j])
                    self.pbest[j] = self.particles[j]
                    self.rbest[j] = values
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)