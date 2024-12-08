import numpy as np
import random

class EPOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.fitness_history = []

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

        # Refine the strategy using Pareto Optimization
        while len(self.elite) < self.elite_size:
            # Select a random elite
            elite = random.sample(self.elite, 1)

            # Evaluate the elite
            fitness_values = [fitness_func(x) for x in elite]
            indices = np.argsort(fitness_values)[:self.population_size]
            elite = [self.population[i] for i in indices]

            # Refine the elite
            while len(elite) < self.elite_size:
                # Select a random individual from the population
                individual = random.sample(self.population, 1)[0]

                # Evaluate the individual
                fitness_values = [fitness_func(x) for x in individual]
                indices = np.argsort(fitness_values)[:self.population_size]
                individual = [self.population[i] for i in indices]

                # Add the individual to the elite
                elite.append(individual)

            # Replace the elite with the refined elite
            self.elite = elite

        return self.elite[0]

# Description: Evolutionary Pareto Optimization using Genetic Algorithm for Black Box Optimization
# Code: 