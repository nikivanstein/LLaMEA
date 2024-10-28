import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        def fitness_func(x):
            return evaluate_func(x)

        while len(self.elite) < self.elite_size:
            # Selection
            fitness_values = [fitness_func(x) for x in self.population]
            indices = np.argsort(fitness_values)[:self.population_size]
            self.elite = [self.population[i] for i in indices]

            # Crossover
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(self.elite, 2)
                child = (parent1 + parent2) / 2
                children.append(child)

            # Mutation
            for child in children:
                if random.random() < 0.1:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        # Evaluate the best individual to refine the strategy
        best_individual = self.elite[0]
        best_fitness = fitness_func(best_individual)
        best_index = np.argmin([fitness_func(x) for x in self.population])
        self.elite[best_index] = np.random.uniform(-5.0, 5.0)

        # Refine the strategy
        new_individual = self.evaluate_fitness(best_individual)
        new_fitness = fitness_func(new_individual)
        if new_fitness < best_fitness * 0.45:
            self.elite[best_index] = new_individual
        else:
            # Revert to the original elite
            self.elite[best_index] = self.elite[best_index]

        return self.elite[0]

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 