import numpy as np

class PSMAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.pbest = np.zeros((budget, dim))
        self.rbest = np.zeros((budget, dim))
        self.memory = {}
        self.mutation_prob = 0.35
        self.mutation_rate = 0.1

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
            # Apply adaptive mutation
            if np.random.rand() < self.mutation_prob:
                mutation_index = np.random.choice(self.particles.shape[0])
                mutated_particle = self.particles[mutation_index] + np.random.uniform(-self.mutation_rate, self.mutation_rate, self.dim)
                self.particles[mutation_index] = mutated_particle
        # Return the best solution found
        return self.rbest[np.argmin(self.rbest, axis=1)]

# Example usage
def func(x):
    return np.sum(x**2)

psmam = PSMAM(budget=10, dim=2)
best_solution = psmam(func)
print(best_solution)