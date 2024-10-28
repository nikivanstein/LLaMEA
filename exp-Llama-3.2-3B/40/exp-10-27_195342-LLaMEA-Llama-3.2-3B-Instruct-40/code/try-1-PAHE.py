import numpy as np
import random
from scipy.optimize import differential_evolution

class PAHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.search_space = (-5.0, 5.0)
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                'params': np.random.uniform(self.search_space[0], self.search_space[1], self.dim),
                'fitness': self.evaluate_func(self.evaluate_func, individual['params'])
            }
            population.append(individual)
        return population

    def evaluate_func(self, func, params):
        return func(params)

    def __call__(self, func):
        for _ in range(self.budget):
            # Select the best individual
            best_individual = max(self.population, key=lambda x: x['fitness'])

            # Refine the strategy of the best individual
            if random.random() < 0.4:
                for i in range(self.dim):
                    new_param = random.uniform(self.search_space[0], self.search_space[1])
                    if random.random() < 0.4:
                        best_individual['params'][i] = new_param
                    else:
                        best_individual['params'][i] = best_individual['params'][i] + random.uniform(-0.1, 0.1)

            # Evaluate the new individual
            best_individual['fitness'] = self.evaluate_func(func, best_individual['params'])

            # Replace the worst individual with the new one
            self.population.remove(min(self.population, key=lambda x: x['fitness']))
            self.population.append(best_individual)

        return min(self.population, key=lambda x: x['fitness'])

# Example usage
def func(x):
    return np.sum(x**2)

bbo = PAHE(50, 5)
best_individual = bbo(func)
print(best_individual['params'])
print(best_individual['fitness'])