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
                self.refine_particles(i)
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

    def refine_particles(self, index):
        # Refine the particles by changing their values with a certain probability
        for _ in range(int(self.budget * self.refine_probability)):
            # Select a particle to refine
            particle_index = np.random.choice(self.budget)
            # Change the value of the particle with a certain probability
            if np.random.rand() < 0.5:
                self.particles[particle_index] += np.random.uniform(-0.5, 0.5, self.dim)

# Example usage
def func(x):
    return np.sum(x**2)

psm = PSM(budget=10, dim=2)
best_solution = psm(func)
print(best_solution)