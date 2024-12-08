import numpy as np
import random

class EvolutionaryParticleSwarmOptimization:
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
        self.adaptive_crossover = np.zeros((self.population_size, self.dim))

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

                # Adaptive crossover
                for i in range(self.num_particles):
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        j = random.randint(0, self.num_particles - 1)
                        k = random.randint(0, self.num_particles - 1)

                        # Crossover
                        child = (particles[i, :] + particles[j, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Update adaptive crossover
                        self.adaptive_crossover[i, :] = child

                        # Replace particle
                        particles[i, :] = child

                # Replace best particle
                self.pbest = np.copy(particles)
                self.gbest = np.copy(particles[0])

            # Update adaptive crossover
            for i in range(self.num_particles):
                if random.random() < self.adaptation_rate:
                    self.adaptive_crossover[i, :] += np.random.uniform(-1.0, 1.0, self.dim)

            # Replace adaptive crossover
            self.adaptive_crossover = np.clip(self.adaptive_crossover, self.lower_bound, self.upper_bound)

            # Select best adaptive crossover particle
            best_adaptive_crossover = np.min(self.adaptive_crossover, axis=0)
            self.adaptive_crossover = np.where(np.all(self.adaptive_crossover == best_adaptive_crossover, axis=1)[:, np.newaxis], self.adaptive_crossover, np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim)))

            # Replace particles with adaptive crossover
            particles = self.adaptive_crossover

        # Return the best solution
        return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = EvolutionaryParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)