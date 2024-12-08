import numpy as np
import random
import math

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
        self.differential_evolution_probability = 0.5
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_history = np.zeros((self.population_size, self.dim))
        self.pbest_history_count = 0
        self.pbest_history_best = np.zeros(self.dim)

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

                # Differential evolution
                for i in range(self.num_particles):
                    if random.random() < self.differential_evolution_probability:
                        # Select three particles
                        j = random.randint(0, self.num_particles - 1)
                        k = random.randint(0, self.num_particles - 1)
                        l = random.randint(0, self.num_particles - 1)

                        # Calculate differential evolution
                        diff = particles[k, :] - particles[j, :]
                        child = particles[i, :] + self.differential_evolution_probability * diff

                        # Replace particle
                        particles[i, :] = child

            # Update pbest history
            if self.pbest_history_count < self.population_size:
                self.pbest_history[self.pbest_history_count, :] = self.pbest[self.pbest_history_count, :]
                self.pbest_history_count += 1
                if np.any(self.pbest[self.pbest_history_count - 1, :] < self.pbest_history[self.pbest_history_count - 1, :]):
                    self.pbest_history_best[:] = self.pbest[self.pbest_history_count - 1, :]
                    self.pbest_history_best_count += 1

            # Check for convergence
            if self.pbest_history_best_count > 10:
                self.pbest_history_best_count = 0

        # Return the best solution
        return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = CrossoverParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)