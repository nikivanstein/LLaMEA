import numpy as np
import random
from deap import base, creator, tools, algorithms

class MultiObjectiveParticleSwarmOptimization:
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
        self.pbest = np.zeros((self.population_size, self.dim, 2))
        self.gbest = np.zeros(self.dim)
        self.individuals = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitnesses = np.zeros(self.population_size)

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            self.individuals = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
            self.fitnesses = np.zeros(self.population_size)
            self.pbest = np.zeros((self.population_size, self.dim, 2))
            self.gbest = np.zeros(self.dim)

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate particles
                for i in range(self.population_size):
                    values = func(self.individuals[i, :])
                    self.fitnesses[i] = values[0]
                    self.pbest[i, :, 0] = self.individuals[i, :]
                    self.pbest[i, :, 1] = values[1]

                # Update pbest and gbest
                for i in range(self.population_size):
                    if self.fitnesses[i] < self.pbest[i, 0, 0]:
                        self.pbest[i, :, 0] = self.individuals[i, :]
                        self.pbest[i, :, 1] = self.fitnesses[i]
                    if self.fitnesses[i] < self.gbest[0]:
                        self.gbest[:] = self.individuals[i, :]

                # Crossover and mutation
                for i in range(self.population_size):
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        j = random.randint(0, self.population_size - 1)
                        k = random.randint(0, self.population_size - 1)

                        # Crossover
                        child = (self.individuals[i, :] + self.individuals[j, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Mutation
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Replace particle
                        self.individuals[i, :] = child

            # Return the best solution
            return self.gbest

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2, x[0]**2 + x[1]**2

optimizer = MultiObjectiveParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)