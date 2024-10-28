import numpy as np
import random

class DifferentialEvolutionWithAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.num_particles = self.population_size
        self.num_iterations = self.budget
        self.crossover_probability = 0.5
        self.adaptation_rate = 0.1
        self.difficulty = 0.5
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)

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

                # Differential evolution
                for i in range(self.num_particles):
                    # Select three random particles
                    j = random.randint(0, self.num_particles - 1)
                    k = random.randint(0, self.num_particles - 1)
                    l = random.randint(0, self.num_particles - 1)

                    # Calculate differences
                    diff1 = particles[j, :] - particles[i, :]
                    diff2 = particles[k, :] - particles[i, :]
                    diff3 = particles[l, :] - particles[i, :]
                    diff = (diff1 + diff2 + diff3) / 3

                    # Calculate new particle
                    new_particle = particles[i, :] + self.difficulty * diff
                    if random.random() < self.crossover_probability:
                        # Select two random particles
                        m = random.randint(0, self.num_particles - 1)
                        n = random.randint(0, self.num_particles - 1)

                        # Crossover
                        child = (new_particle + particles[m, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Mutation
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Replace particle
                        particles[i, :] = child

        # Return the best solution
        return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = DifferentialEvolutionWithAdaptiveCrossover(budget=100, dim=2)
result = optimizer(func)
print(result)