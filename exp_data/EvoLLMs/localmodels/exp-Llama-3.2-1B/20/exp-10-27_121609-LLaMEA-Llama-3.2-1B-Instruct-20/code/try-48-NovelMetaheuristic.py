# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize

class NovelMetaheuristic:
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
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                return individual
            else:
                return individual

        def mutate(individual):
            new_individual = individual + np.random.uniform(-1.0, 1.0, self.dim)
            return new_individual

        def mutate_new(individual):
            new_individual = mutate(individual)
            if np.random.rand() < 0.2:
                new_individual = mutate(new_individual)
            return new_individual

        def mutate_all(individual):
            new_individual = mutate(individual)
            for _ in range(self.dim):
                new_individual = mutate(new_individual)
            return new_individual

        def next_generation():
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                new_individual = evaluate_fitness(self.population[i])
                new_population[i] = new_individual
                if np.random.rand() < 0.2:
                    new_individual = mutate_new(new_individual)
                elif np.random.rand() < 0.2:
                    new_individual = mutate_all(new_individual)
                else:
                    new_individual = self.population[i]
            return new_population

        next_generation = next_generation()
        return next_generation

nneo = NovelMetaheuristic(100, 10)