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
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
            self.pbest = np.copy(self.population)
            self.gbest = np.copy(self.population[0])

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate particles
                values = func(self.population)

                # Update pbest and gbest
                for i in range(self.population_size):
                    if values[i] < self.pbest[i, 0]:
                        self.pbest[i, :] = self.population[i, :]
                    if values[i] < self.gbest[0]:
                        self.gbest[:] = self.population[i, :]

                # Crossover and mutation
                for i in range(self.population_size):
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        j = random.randint(0, self.population_size - 1)
                        k = random.randint(0, self.population_size - 1)

                        # Crossover
                        child = (self.population[i, :] + self.population[j, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Mutation
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Replace particle
                        self.population[i, :] = child

            # Update the best solution
            self.gbest = np.copy(self.population[np.argmin(np.min(self.population, axis=0))])

        # Return the best solution
        return self.gbest

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = CrossoverParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)