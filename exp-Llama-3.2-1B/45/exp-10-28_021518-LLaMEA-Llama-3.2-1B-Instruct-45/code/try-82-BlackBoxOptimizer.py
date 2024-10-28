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

        # Perform a single evaluation of the elite to get the best individual
        best_individual = self.elite[0]
        best_fitness = fitness_func(best_individual)
        best_x = best_individual

        # Refine the strategy using a probability of 0.45
        if random.random() < 0.45:
            # Perturb the best individual
            perturbed_x = best_x + np.random.uniform(-1.0, 1.0)
            # Evaluate the perturbed individual
            perturbed_fitness = fitness_func(perturbed_x)
            # Update the best individual if the perturbed individual is better
            if perturbed_fitness < best_fitness:
                best_individual = perturbed_x
                best_fitness = perturbed_fitness

        return best_individual

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 