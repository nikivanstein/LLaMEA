import numpy as np
import random
import operator

class MultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.particles = []
        self.best_particles = []
        self.crossover_probability = 0.5
        self.mutation_probability = 0.1
        self.probability = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            for _ in range(self.swarm_size):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                self.particles.append(particle)

            for i in range(self.swarm_size):
                particle = self.particles[i]
                # Evaluate the function
                fitness = func(particle)

                # Update the best particle
                if fitness < func(self.best_particles[i]):
                    self.best_particles[i] = particle

                # Update the particle
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                if r1 < self.crossover_probability:
                    # Crossover
                    new_particle = np.array([
                        particle[0] + np.random.uniform(-1, 1),
                        particle[1] + np.random.uniform(-1, 1)
                    ])
                    if np.random.rand() < self.probability:
                        self.particles[i] = new_particle
                if r2 < self.mutation_probability:
                    # Mutation
                    mutation_factor = np.random.uniform(-0.1, 0.1, self.dim)
                    self.particles[i] = self.particles[i] + mutation_factor

            # Update the best particles
            for i in range(self.swarm_size):
                self.best_particles[i] = self.particles[i]

        # Return the best particle
        return min(self.best_particles, key=func)

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = MultiSwarmPSO(budget, dim)
best_solution = optimizer(func)
print("Best solution:", best_solution)