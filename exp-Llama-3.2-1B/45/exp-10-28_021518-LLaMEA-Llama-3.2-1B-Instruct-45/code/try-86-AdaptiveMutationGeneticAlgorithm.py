import random
import numpy as np

class AdaptiveMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.mutation_rate = 0.1

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
                    if np.random.rand() < self.mutation_rate:
                        index = random.randint(0, self.dim - 1)
                        child[index] -= random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        # Refine the strategy using adaptive mutation
        for individual in self.elite:
            updated_individual = individual
            for _ in range(self.budget):
                fitness_values = [fitness_func(updated_individual)]
                indices = np.argsort(fitness_values)[:self.population_size]
                updated_individual = self.elite[indices]
            updated_individual = updated_individual + random.uniform(-1.0, 1.0) * 0.5
            self.elite = [updated_individual]
        return self.elite[0]

# Description: Evolutionary Algorithm for Black Box Optimization using Adaptive Mutation
# Code: 