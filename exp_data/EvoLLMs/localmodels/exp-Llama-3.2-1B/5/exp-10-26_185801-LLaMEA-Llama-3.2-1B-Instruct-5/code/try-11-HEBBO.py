import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        """Randomly mutate the individual by adding or subtracting a random value from the search space."""
        new_individual = individual.copy()
        if random.random() < 0.5:
            if random.random() < 0.5:
                new_individual += np.random.uniform(-1.0, 1.0, self.dim)
            else:
                new_individual -= np.random.uniform(-1.0, 1.0, self.dim)
        return new_individual

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create a new offspring."""
        if len(parent1) > len(parent2):
            parent1, parent2 = parent2, parent1
        offspring = parent1[:len(parent2)]
        for i in range(len(parent2), len(offspring)):
            if random.random() < 0.5:
                offspring[i] = parent2[i]
        return offspring

    def select(self, parents, num_parents):
        """Select parents based on their fitness scores."""
        fitness_scores = [p.fitness for p in parents]
        selected_parents = random.choices(parents, weights=fitness_scores, k=num_parents)
        return selected_parents

    def __call__(self, func):
        # Initialize the population with random individuals
        population = [self.evaluate_individual(func) for _ in range(100)]

        # Evolve the population using mutation and crossover
        for _ in range(100):
            population = [self.evaluate_individual(func) for func in population]
            population = self.select(population, 10)
            population = [self.mutate(individual) for individual in population]

        # Find the individual with the best fitness score
        best_individual = max(population, key=self.f)

        return best_individual

# One-line description: Adaptive Black Box Optimization using Evolutionary Strategies