import numpy as np
import random

class CrossoverParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.num_particles = self.population_size
        self.num_iterations = self.budget
        self.crossover_probability = 0.8
        self.adaptation_rate = 0.1
        self.mutation_probability = 0.2
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.crossover_mask = np.ones((self.population_size, self.num_particles), dtype=bool)
        self.mutation_mask = np.ones((self.population_size, self.num_particles), dtype=bool)

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
            self.pbest = np.copy(particles)
            self.gbest = np.copy(particles[0])

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate particles
                values = func(particles)

                # Update pbest and gbest
                for i in range(self.num_particles):
                    if values[i] < self.pbest[i, 0]:
                        self.pbest[i, :] = particles[i, :]
                    if values[i] < self.gbest[0]:
                        self.gbest[:] = particles[i, :]

                # Crossover and mutation
                for i in range(self.population_size):
                    for j in range(self.num_particles):
                        if self.crossover_mask[i, j]:
                            # Select two particles
                            k = random.randint(0, self.num_particles - 1)
                            l = random.randint(0, self.num_particles - 1)

                            # Crossover
                            child = (particles[i, :] + particles[k, :]) / 2
                            if random.random() < self.adaptation_rate:
                                child += np.random.uniform(-1.0, 1.0, self.dim)
                            if random.random() < self.mutation_probability:
                                child += np.random.uniform(-1.0, 1.0, self.dim)

                            # Replace particle
                            particles[i, :] = child

                            # Update crossover mask
                            self.crossover_mask[i, j] = False
                            self.crossover_mask[i, k] = False

                        if self.mutation_mask[i, j]:
                            # Mutation
                            child = particles[i, :]
                            if random.random() < self.adaptation_rate:
                                child += np.random.uniform(-1.0, 1.0, self.dim)
                            if random.random() < self.mutation_probability:
                                child += np.random.uniform(-1.0, 1.0, self.dim)

                            # Replace particle
                            particles[i, :] = child

                            # Update mutation mask
                            self.mutation_mask[i, j] = False

            # Return the best solution
            return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = CrossoverParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)