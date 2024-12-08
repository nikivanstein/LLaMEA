import random
import numpy as np

class EvolutionaryOptimizer:
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

            # Evolution Strategies
            # 1. Evolutionary Crossover
            children = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(self.elite, 2)
                child = (parent1 + parent2) / 2
                children.append(child)

            # 2. Evolutionary Mutation
            for child in children:
                if random.random() < 0.1:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        # 3. Evolutionary Inversion
        inversion_indices = []
        for i in range(len(self.elite)):
            inversion_indices.append(i)
            inversion_indices.append(len(self.elite) - i - 1)
        inversion_indices = np.array(inversion_indices)
        self.elite = np.array([self.elite[i] for i in inversion_indices])

        # 4. Evolutionary Normalization
        self.elite = np.clip(self.elite, -5.0, 5.0)

        return self.elite[0]

# Description: Evolutionary Algorithm for Black Box Optimization using Evolution Strategies
# Code: 