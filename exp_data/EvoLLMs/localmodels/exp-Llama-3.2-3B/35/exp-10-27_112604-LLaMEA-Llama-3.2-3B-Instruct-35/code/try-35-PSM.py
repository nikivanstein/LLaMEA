import numpy as np

class PSM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.refine_probability = 0.35

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
            # Refine the strategy with a probability of 0.35
            if np.random.rand() < self.refine_probability:
                # Generate a new particle by perturbing the current particle
                new_particle = self.particles[i] + np.random.uniform(-0.5, 0.5, self.dim)
                # Evaluate the function at the new particle
                new_values = func(new_particle)
                # Update the personal best and global best if the new values are better
                if np.any(new_values < self.rbest, axis=1):
                    self.pbest[i] = new_particle
                    self.rbest[np.argmin(self.rbest, axis=1)] = new_values
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)