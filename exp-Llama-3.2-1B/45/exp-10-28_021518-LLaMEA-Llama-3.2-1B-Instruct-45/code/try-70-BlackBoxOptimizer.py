import random
import numpy as np
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.population_history = []

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

        # Update the elite with the new population
        new_population = self.generate_population()
        new_population = [copy.deepcopy(individual) for individual in new_population]
        for i in range(len(self.elite)):
            new_population[i] = self.evaluate_func(new_population[i])

        # Replace the elite with the new population
        self.elite = new_population

        # Update the population history
        self.population_history.append(len(self.elite))

        return self.elite[0]

    def update_elite(self, budget):
        # Update the elite with the new population
        new_population = self.generate_population()
        new_population = [copy.deepcopy(individual) for individual in new_population]
        for i in range(len(self.elite)):
            new_population[i] = self.evaluate_func(new_population[i])

        # Replace the elite with the new population
        self.elite = new_population

        # Update the population history
        self.population_history.append(len(self.elite))

        # Refine the elite using the probability 0.45
        refined_elite = []
        for _ in range(len(self.elite)):
            if random.random() < 0.45:
                refined_elite.append(self.elite.pop(0))

        # Replace the elite with the refined elite
        self.elite = refined_elite

        # Update the population history
        self.population_history.append(len(self.elite))

        return self.elite[0]