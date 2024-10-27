import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.01, crossover_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        def func_eval(x):
            return func(x)

        # Initialize the population with random points in the search space
        x = np.random.uniform(-5.0, 5.0, self.dim)
        population = [x] * self.population_size

        # Evaluate the function for each point in the population
        for _ in range(self.budget):
            # Select the fittest points to reproduce
            fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

            # Create new offspring by crossover and mutation
            offspring = []
            for i in range(self.population_size // 2):
                parent1, parent2 = random.sample(fittest_points, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child += random.uniform(-5.0, 5.0)
                offspring.append(child)

            # Replace the worst points in the population with the new offspring
            population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Select the fittest points to reproduce
        fittest_points = sorted(population, key=func_eval, reverse=True)[:self.population_size // 2]

        # Create new offspring by crossover and mutation
        offspring = []
        for i in range(self.population_size // 2):
            parent1, parent2 = random.sample(fittest_points, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child += random.uniform(-5.0, 5.0)
            offspring.append(child)

        # Replace the worst points in the population with the new offspring
        population = [x if func_eval(x) < func_eval(p) else p for p in population]

        # Update the best individual and fitness
        if func_eval(self.best_individual) > func_eval(population[0]):
            self.best_individual = population[0]
            self.best_fitness = func_eval(self.best_individual)

        # Evaluate the best individual
        updated_individual = self.evaluate_fitness(self.best_individual)

        # If the best fitness is worse than the current best fitness, change the best individual
        if updated_individual > self.best_fitness:
            self.best_individual = updated_individual
            self.best_fitness = updated_individual

        return updated_individual

    def evaluate_fitness(self, individual):
        return func_eval(individual)

# One-line description: 
# An adaptive evolutionary optimization algorithm for black box functions that balances exploration and exploitation using a population-based approach and evolutionary mutation.