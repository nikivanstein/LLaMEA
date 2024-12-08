# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 
import random
import numpy as np
from scipy.optimize import minimize
from collections import deque

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

        # Run the elite to refine its strategy
        new_individuals = []
        for _ in range(self.budget):
            new_individual = self.evaluate_fitness(new_individual)
            new_individuals.append(new_individual)
        new_individuals = np.array(new_individuals)
        new_individuals = (new_individuals - new_individuals.min()) / (new_individuals.max() - new_individuals.min())
        new_individuals = new_individuals * (self.budget - 1) / self.budget
        new_individuals = new_individuals.astype(np.float32)
        new_individuals = new_individuals.tolist()
        self.elite = new_individuals

        return self.elite[0]

    def evaluate_fitness(self, func):
        return minimize(func, np.array([0.0]), method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim, options={"maxiter": 100})

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 