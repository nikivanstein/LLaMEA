import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.adaptive Mutation = True
        self.adaptiveMutationThreshold = 0.1

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
            if self.adaptiveMutation:
                # Adaptive mutation strategy: change the mutation probability based on the fitness values
                if np.mean(fitness_values) < self.adaptiveMutationThreshold:
                    mutation_probability = 0.5
                else:
                    mutation_probability = 0.1
                for child in children:
                    if random.random() < mutation_probability:
                        index = random.randint(0, self.dim - 1)
                        child[index] += random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        return self.elite[0]

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 