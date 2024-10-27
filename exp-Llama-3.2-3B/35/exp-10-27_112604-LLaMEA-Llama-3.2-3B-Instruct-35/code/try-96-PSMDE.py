import numpy as np

class PSMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.de = 0.5

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
            # Apply differential evolution
            for j in range(self.dim):
                diff = np.random.uniform(-1, 1, (self.budget,))
                new_particle = self.particles[i] + self.de * diff
                values = func(new_particle)
                if np.any(values < self.rbest, axis=1):
                    self.memory[i] = values
                    self.rbest[np.argmin(self.rbest, axis=1)] = values
                    self.particles[i] = new_particle
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psmde = PSMDE(budget=10, dim=2)
best_solution = psmde(func)
print(best_solution)