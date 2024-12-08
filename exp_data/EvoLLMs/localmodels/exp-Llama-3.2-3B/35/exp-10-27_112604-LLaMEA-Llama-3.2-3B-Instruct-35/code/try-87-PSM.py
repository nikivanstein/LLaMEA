import numpy as np

class PSM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.adaptive_lr = 0.5

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
            # Update the particles using the PSO update rule with adaptive learning rate
            self.particles[i] = self.particles[i] + self.adaptive_lr * (self.pbest[i] - self.particles[i]) + self.adaptive_lr * (self.rbest - self.particles[i])
            # Refine the strategy with 35% probability
            if np.random.rand() < 0.35:
                # Get the best individual from the memory
                best_individual = np.argmin(self.rbest, axis=1)
                # Update the particles with the best individual
                self.particles[i] = self.particles[best_individual]
                # Update the personal best and global best
                self.pbest[i] = self.particles[i]
                self.rbest[i] = func(self.particles[i])
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)