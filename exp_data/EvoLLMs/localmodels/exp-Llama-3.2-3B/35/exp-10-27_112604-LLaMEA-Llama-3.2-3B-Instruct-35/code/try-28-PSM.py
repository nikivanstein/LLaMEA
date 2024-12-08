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
            # Refine the strategy with 35% probability
            if np.random.rand() < self.refine_probability:
                # Randomly select a particle to refine
                refine_particle_index = np.random.randint(0, self.budget)
                # Refine the particle's position using the PSO update rule
                refine_particle = self.particles[refine_particle_index]
                refine_particle = refine_particle + 0.5 * (self.pbest[refine_particle_index] - refine_particle) + 0.5 * (self.rbest - refine_particle)
                # Replace the particle's position with the refined position
                self.particles[refine_particle_index] = refine_particle
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)