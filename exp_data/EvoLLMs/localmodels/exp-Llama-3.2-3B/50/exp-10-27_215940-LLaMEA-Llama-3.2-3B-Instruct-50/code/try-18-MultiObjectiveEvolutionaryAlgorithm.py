import numpy as np
import random
import operator

class MultiObjectiveEvolutionaryAlgorithm:
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
        self.mutate_probability = 0.2
        self.mutation_range = 1.0
        self.fitness_values = np.zeros((self.num_particles, self.dim))
        self.pbest = np.zeros((self.num_particles, self.dim))
        self.gbest = np.zeros(self.dim)

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
            self.fitness_values = np.zeros((self.num_particles, self.dim))
            self.pbest = np.copy(particles)
            self.gbest = np.copy(particles[0])

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate particles
                values = func(particles)

                # Update fitness values
                for i in range(self.num_particles):
                    for j in range(self.dim):
                        self.fitness_values[i, j] = values[i, j]

                # Update pbest and gbest
                for i in range(self.num_particles):
                    for j in range(self.dim):
                        if self.fitness_values[i, j] < self.pbest[i, j]:
                            self.pbest[i, :] = particles[i, :]
                        if self.fitness_values[i, j] < self.gbest[0]:
                            self.gbest[:] = particles[i, :]

                # Crossover and mutation
                for i in range(self.num_particles):
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        j = random.randint(0, self.num_particles - 1)
                        k = random.randint(0, self.num_particles - 1)

                        # Crossover
                        child = (particles[i, :] + particles[j, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Mutation
                        if random.random() < self.mutate_probability:
                            child += np.random.uniform(-self.mutation_range, self.mutation_range, self.dim)

                        # Replace particle
                        particles[i, :] = child

            # Return the best solution
            return self.gbest

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = MultiObjectiveEvolutionaryAlgorithm(budget=100, dim=2)
result = optimizer(func)
print(result)