import numpy as np
import random

class MultiSwarmQIO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.particles = []
        self.best_particles = []
        self.crossover_probability = 0.4  # Increased crossover probability
        self.mutation_probability = 0.3  # Decreased mutation probability
        self.crossover_mutation_probability = 0.3  # Decreased crossover-mutation probability
        self.quantum_bit = 0.3  # Decreased quantum bit probability
        self.pbest = np.full((self.swarm_size, self.dim), -np.inf)
        self.gbest = np.full(self.dim, -np.inf)
        self.count = 0

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
                if fitness < self.pbest[i, 0]:
                    self.pbest[i, :] = particle
                    self.gbest = np.copy(particle)

                # Update the particle
                if random.random() < self.crossover_probability:
                    # Crossover
                    new_particle = np.copy(particle)
                    for j in range(self.dim):
                        if random.random() < 0.7:
                            new_particle[j] += np.random.uniform(-1, 1)
                    self.particles[i] = new_particle

                if random.random() < self.mutation_probability:
                    # Mutation
                    self.particles[i] += np.random.uniform(-0.05, 0.05, self.dim)
                    # Apply quantum bit flip
                    for j in range(self.dim):
                        if random.random() < self.quantum_bit:
                            self.particles[i][j] *= -1

                if random.random() < self.crossover_mutation_probability:
                    # Crossover-mutation hybrid
                    new_particle = np.copy(particle)
                    for j in range(self.dim):
                        if random.random() < 0.8:
                            new_particle[j] += np.random.uniform(-1, 1)
                    self.particles[i] = new_particle
                    self.particles[i] += np.random.uniform(-0.05, 0.05, self.dim)
                    # Apply quantum bit flip
                    for j in range(self.dim):
                        if random.random() < self.quantum_bit:
                            self.particles[i][j] *= -1

                # Update the best particles
                for j in range(self.dim):
                    if self.pbest[i, j] > self.gbest[j]:
                        self.pbest[i, j] = self.gbest[j]

            # Update the best particles
            for i in range(self.swarm_size):
                self.best_particles.append(self.pbest[i, :])

        # Return the best particle
        return self.best_particles[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = MultiSwarmQIO(budget, dim)
best_solution = optimizer(func)
print("Best solution:", best_solution)