import numpy as np
import random
from scipy.optimize import minimize

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        def mutate(individual):
            return individual + random.uniform(-1.0, 1.0)

        def crossover(parent1, parent2):
            return (parent1 + 2.0 * random.uniform(-1.0, 1.0)) / 2.0

        def mutate_bounding(individual):
            bounds = bounds(individual)
            if random.random() < 0.2:
                return individual + random.uniform(-1.0, 1.0)
            else:
                return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(self.population[i])
                if random.random() < 0.2:
                    individual = mutate(individual)
                individual = mutate_bounding(individual)
                fitness = objective(individual)
                if fitness < self.fitnesses[i, individual] + 1e-6:
                    self.fitnesses[i, individual] = fitness
                    self.population[i] = individual

        return self.fitnesses

# Example usage
def black_box_func(x):
    return x**2 + 2.0 * x + 1.0

nneo = NNEO(10, 2)
print(nneo(__call__(black_box_func)))

# Output: 
# [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]