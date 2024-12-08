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
        self.crossover_probability = 0.5
        self.adaptation_rate = 0.1
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_history = np.zeros((self.population_size, self.budget, self.dim))
        self.gbest_history = np.zeros((self.dim, self.budget))

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
            self.pbest_history[:, :, :] = np.copy(particles)
            self.gbest_history[:, :] = np.copy(particles[0])

            # Main loop
            for iteration in range(self.num_iterations):
                # Evaluate particles
                values = func(particles)

                # Update pbest and gbest
                for i in range(self.num_particles):
                    if values[i] < self.pbest[i, iteration]:
                        self.pbest[i, iteration] = values[i]
                        self.pbest_history[i, iteration, :] = particles[i, :]
                    if values[i] < self.gbest[iteration]:
                        self.gbest[iteration] = values[i]
                        self.gbest_history[iteration, :] = particles[i, :]

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
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Replace particle
                        particles[i, :] = child

        # Return the best solution
        return self.gbest_history[-1, :]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = CrossoverParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)