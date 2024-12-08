import numpy as np
import random
from scipy.optimize import minimize

class EvolutionaryGradientBoosting:
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
        self.gradient_step_size = 0.1
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize population
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate population
                values = func(self.population)

                # Update population
                for i in range(self.num_particles):
                    # Crossover
                    if random.random() < self.crossover_probability:
                        # Select two particles
                        j = random.randint(0, self.num_particles - 1)
                        k = random.randint(0, self.num_particles - 1)

                        # Crossover
                        child = (self.population[i, :] + self.population[j, :]) / 2
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Mutation
                        if random.random() < self.adaptation_rate:
                            child += np.random.uniform(-1.0, 1.0, self.dim)

                        # Replace particle
                        self.population[i, :] = child

                # Gradient-based optimization
                for i in range(self.num_particles):
                    # Evaluate function
                    values = func(self.population[i, :])

                    # Update particle
                    self.population[i, :] -= self.gradient_step_size * np.array(func(self.population[i, :]).grad)

                    # Ensure bounds
                    self.population[i, :] = np.clip(self.population[i, :], self.lower_bound, self.upper_bound)

        # Return the best solution
        return self.population[np.argmin(np.array([func(x) for x in self.population]))]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = EvolutionaryGradientBoosting(budget=100, dim=2)
result = optimizer(func)
print(result)