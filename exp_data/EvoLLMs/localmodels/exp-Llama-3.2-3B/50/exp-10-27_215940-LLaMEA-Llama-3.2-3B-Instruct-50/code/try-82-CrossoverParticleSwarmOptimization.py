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
        self.differential_evolution_probability = 0.2
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_history = np.zeros((self.population_size, self.num_iterations, self.dim))
        self.gbest_history = np.zeros((self.dim, self.num_iterations))

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize particles
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
            self.pbest_history[:, 0, :] = particles
            self.gbest_history[:, 0] = particles[0]

            # Main loop
            for i in range(self.num_iterations - 1):
                # Evaluate particles
                values = func(particles)

                # Update pbest and gbest
                for j in range(self.num_particles):
                    if values[j] < self.pbest[j, 0]:
                        self.pbest_history[j, i + 1, :] = particles[j, :]
                        self.pbest[j, :] = particles[j, :]
                    if values[j] < self.gbest[0]:
                        self.gbest_history[:, i + 1] = particles[j, :]
                        self.gbest[:] = particles[j, :]

                # Crossover and mutation
                for j in range(self.num_particles):
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        k = random.randint(0, self.num_particles - 1)
                        l = random.randint(0, self.num_particles - 1)

                        # Crossover
                        child = (particles[j, :] + particles[k, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Mutation
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Replace particle
                        particles[j, :] = child

                # Differential evolution
                if random.random() < self.differential_evolution_probability:
                    # Select three particles
                    m = random.randint(0, self.num_particles - 1)
                    n = random.randint(0, self.num_particles - 1)
                    o = random.randint(0, self.num_particles - 1)

                    # Calculate differential evolution parameters
                    r1 = random.random()
                    r2 = random.random()

                    # Calculate new particle
                    new_particle = particles[m, :] + r1 * (particles[n, :] - particles[m, :])
                    new_particle = new_particle + r2 * (particles[o, :] - particles[m, :])

                    # Replace particle
                    particles[m, :] = new_particle

            # Return the best solution
            return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = CrossoverParticleSwarmOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)