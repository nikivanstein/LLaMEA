import numpy as np
import random

class NPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_particles = np.copy(self.particles)
        self.best_values = np.full(self.population_size, np.inf)
        self.cognitive_coefficient = 0.5
        self.social_coefficient = 0.5
        self.mutation_probability = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                # Evaluate the function at the current particle position
                value = func(self.particles[i])

                # Update the best particle if the current value is better
                if value < self.best_values[i]:
                    self.best_particles[i] = self.particles[i]
                    self.best_values[i] = value

                # Update the cognitive and social components
                self.particles[i] += self.cognitive_coefficient * (self.best_particles[i] - self.particles[i]) + \
                                      self.social_coefficient * (self.particles[np.random.choice(self.population_size, 1, p=self.best_values / np.sum(self.best_values))][0] - self.particles[i])

                # Apply mutation with probability
                if random.random() < self.mutation_probability:
                    self.particles[i] += np.random.uniform(-0.1, 0.1, self.dim)

# Example usage
def func(x):
    return np.sum(x**2)

npso = NPSO(100, 5)
best_value = npso(func)
print(f"Best value: {best_value}")