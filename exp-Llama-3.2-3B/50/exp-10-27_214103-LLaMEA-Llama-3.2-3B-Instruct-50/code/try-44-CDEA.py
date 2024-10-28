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

            # Perform probability-based crossover and mutation
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                if random.random() < 0.5:
                    # Perform single-point crossover
                    parent1, parent2 = random.sample(self.population, 2)
                    child = np.concatenate((parent1, parent2[1:]))
                    new_population[i] = child
                else:
                    # Perform Gaussian mutation
                    mutated_population = new_population[i] + np.random.normal(0, 1, self.dim)
                    new_population[i] = np.clip(mutated_population, -5.0, 5.0)
            self.population = new_population

# Usage:
# bbo_test_suite = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23"]
# for func_name in bbo_test_suite:
#     def func(x):
#         # Define your black box function here
#         return x[0]**2 + x[1]**2 + x[2]**2
#     cdea = CDEA(100, 3)
#     cdea(func)
#     print(f"Function: {func_name}, Score: {cdea.population[0]}")