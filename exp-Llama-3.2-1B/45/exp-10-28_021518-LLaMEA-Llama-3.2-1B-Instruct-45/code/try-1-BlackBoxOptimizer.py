import numpy as np
import random
import matplotlib.pyplot as plt

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.fitness_history = np.zeros((self.population_size, self.dim))

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

            # Update fitness history
            self.fitness_history = np.vstack((self.fitness_history, [fitness_func(x) for x in self.elite]))

        # Refine the elite using the fitness history
        new_individuals = []
        for _ in range(self.population_size):
            fitness_values = [self.fitness_history[i] for i in range(self.dim)]
            indices = np.argsort(fitness_values)[:self.population_size]
            new_individual = [self.elite[i] for i in indices]
            new_individual = (new_individual + 0.1 * [random.uniform(-1.0, 1.0)] * self.dim) / 2
            new_individuals.append(new_individual)

        self.elite = new_individuals

        # Evaluate the new elite
        new_fitness_values = [fitness_func(x) for x in self.elite]
        indices = np.argsort(new_fitness_values)[:self.population_size]
        new_elite = [self.elite[i] for i in indices]

        return new_elite

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 