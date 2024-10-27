import random
import numpy as np

class TreeStructuredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.particles = self._initialize_particles()
        self.fitness_history = []

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def _initialize_particles(self):
        particles = []
        for _ in range(100):
            particle = {}
            for i in range(self.dim):
                particle[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
            particles.append(particle)
        return particles

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_update_particles(func)

    def _evaluate_and_update_particles(self, func):
        fitness = [func(particle) for particle in self.particles]
        self.fitness_history.append(fitness)

        for i, particle in enumerate(self.particles):
            if fitness[i] == 0:
                return  # termination condition

            # Update particle's position
            for j in range(self.dim):
                if random.random() < 0.35:
                    particle[j]['value'] += random.uniform(-1.0, 1.0)
                    if particle[j]['value'] < particle[j]['lower']:
                        particle[j]['value'] = particle[j]['lower']
                    elif particle[j]['value'] > particle[j]['upper']:
                        particle[j]['value'] = particle[j]['upper']

            # Update particle's velocity
            velocity = random.uniform(-1.0, 1.0)
            particle['velocity'] = velocity

            # Update particle's position
            particle['position'] = particle['value']

            # Update particle's fitness
            particle['fitness'] = func(particle)

    def get_particles(self):
        return self.particles

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = TreeStructuredPSO(budget, dim)
evolution()
particles = evolution.get_particles()
for particle in particles:
    print(particle)