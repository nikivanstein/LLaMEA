# Code: 
import random
import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim, elite_size):
        self.budget = budget
        self.dim = dim
        self.elite_size = elite_size
        self.population = self.generate_population()
        self.elite = self.population[:self.elite_size]
        self.alpha = 0.45
        self.beta = 0.55

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
                if random.random() < self.alpha:
                    index = random.randint(0, self.dim - 1)
                    child[index] += random.uniform(-1.0, 1.0)
                if random.random() < self.beta:
                    index = random.randint(0, self.dim - 1)
                    child[index] -= random.uniform(-1.0, 1.0)

            # Replace the elite with the children
            self.elite = children

        # Update strategy
        if len(self.elite) < self.budget:
            new_individual = self.evaluate_fitness(self.elite[0])
            self.elite[0] = new_individual
        else:
            new_individual = self.elite[0]
            for i in range(len(self.elite)):
                if random.random() < self.alpha:
                    index = random.randint(0, self.dim - 1)
                    new_individual[index] += random.uniform(-1.0, 1.0)
                if random.random() < self.beta:
                    index = random.randint(0, self.dim - 1)
                    new_individual[index] -= random.uniform(-1.0, 1.0)

        return new_individual

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 