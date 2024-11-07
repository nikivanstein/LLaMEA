import numpy as np
import random

class HyperParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = []
        self.best_particles = []
        self.crossover_rate = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize particle with random position and velocity
            particle = np.random.uniform(-5.0, 5.0, self.dim)
            velocity = np.random.uniform(-5.0, 5.0, self.dim)

            # Add particle to population
            self.particles.append(particle)

            # Evaluate particle's fitness
            fitness = func(particle)

            # Update particle's best position and velocity
            if fitness > func(self.best_particles[0]):
                self.best_particles.append(particle)
            else:
                self.best_particles[-1] = particle

            # Update particle's velocity and position
            for _ in range(10):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                if r1 < self.crossover_rate:
                    # Perform crossover with a random particle
                    parent1 = random.choice(self.particles)
                    parent2 = random.choice(self.best_particles)
                    child = parent1 + parent2 - parent1 * parent2
                    self.particles.append(child)
                else:
                    # Perform mutation
                    self.particles[-1] += velocity

            # Evaluate particle's fitness again
            fitness = func(self.particles[-1])

            # Update particle's best position and velocity
            if fitness > func(self.best_particles[0]):
                self.best_particles.append(self.particles[-1])
            else:
                self.best_particles[-1] = self.particles[-1]

    def refine_strategy(self):
        # Refine strategy by adding a local search component
        for _ in range(10):
            # Select a random particle
            particle = random.choice(self.particles)

            # Perform a local search around the particle
            for i in range(self.dim):
                # Evaluate the particle's fitness at the current position
                fitness = func(particle)

                # Evaluate the particle's fitness at the neighboring positions
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        neighbor = particle.copy()
                        neighbor[i] += j * 0.1
                        neighbor[i + self.dim] += k * 0.1
                        fitness_neighbor = func(neighbor)

                        # Update the particle's position if the neighboring position has a better fitness
                        if fitness_neighbor < fitness:
                            particle = neighbor

                # Update the particle's fitness
                fitness = func(particle)

            # Update the particle's best position and velocity
            if fitness > func(self.best_particles[0]):
                self.best_particles.append(particle)
            else:
                self.best_particles[-1] = particle