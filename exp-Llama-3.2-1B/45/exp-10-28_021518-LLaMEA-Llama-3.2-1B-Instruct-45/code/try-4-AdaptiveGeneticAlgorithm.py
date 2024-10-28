# Adaptive Genetic Algorithm for Black Box Optimization
# Code: 
import random
import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.elite_size = 10
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.fitness_history = np.zeros((self.population_size, self.dim))
        self.fitness_values = self.evaluate_fitness(self.elite)

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim
            for _ in range(dim):
                population.append(np.random.uniform(-5.0, 5.0))
        return population

    def evaluate_fitness(self, individual):
        def evaluate_func(x):
            return individual[0]

        def fitness_func(x):
            return evaluate_func(x)

        fitness_values = [fitness_func(x) for x in self.fitness_values]
        fitness_values = np.array(fitness_values)
        fitness_values /= fitness_values.sum()
        fitness_values = np.repeat(fitness_values, self.dim)
        return fitness_values

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

        # Adaptive strategy
        new_individual = self.evaluate_fitness(self.elite[0])
        if random.random() < 0.45:
            new_individual = self.evaluate_fitness(self.elite[1])
        return new_individual

# Description: Adaptive Genetic Algorithm for Black Box Optimization
# Code: 