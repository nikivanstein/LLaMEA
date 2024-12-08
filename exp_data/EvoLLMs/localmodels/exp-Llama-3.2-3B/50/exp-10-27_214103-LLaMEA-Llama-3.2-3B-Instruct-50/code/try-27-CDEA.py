import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crowd_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.crowd = np.random.uniform(-5.0, 5.0, (self.crowd_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            population_values = np.array([func(x) for x in self.population])

            # Evaluate the crowd
            crowd_values = np.array([func(x) for x in self.crowd])

            # Select the best individuals
            best_indices = np.argsort(population_values)[:, -self.crowd_size:]
            best_crowd_values = crowd_values[best_indices]

            # Select the worst individuals
            worst_indices = np.argsort(population_values)[:, :self.crowd_size]
            worst_population_values = population_values[worst_indices]

            # Update the population
            self.population = np.concatenate((best_crowd_values, worst_population_values))

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform crossover and mutation with probabilistic refinement
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                if random.random() < 0.5:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = np.concatenate((parent1, parent2[1:]))
                    new_population[i] = child
                else:
                    new_population[i] = self.population[i]
            self.population = new_population

            # Perform Gaussian mutation with probability 0.5
            mutated_population = np.copy(self.population)
            for i in range(self.population_size):
                if random.random() < 0.5:
                    mutated_population[i] += np.random.normal(0, 1, self.dim)
                else:
                    mutated_population[i] = self.population[i]
            self.population = np.clip(mutated_population, -5.0, 5.0)

# Usage
def bbb_optimize(func, budget, dim):
    cdea = CDEA(budget, dim)
    cdea()
    return func

# Example usage
def func(x):
    return np.sum(x**2)

bbob_scores = np.zeros(24)
for i in range(24):
    bbob_scores[i] = bbb_optimize(func, 50, 10)

print(bbob_scores)