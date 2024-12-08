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

class HSBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.selection_prob = 0.5
        self.crossover_prob = 0.5
        self.mutation_prob = 0.1
        self.population_size = 100

    def __call__(self, func):
        # Evaluate the function for each individual in the population
        evaluations = np.random.rand(self.population_size, self.dim)
        evaluations = np.minimum(evaluations, self.search_space)
        evaluations = np.maximum(evaluations, -5.0)
        evaluations = np.minimum(evaluations, 5.0)

        # Select individuals for the next generation
        selected_indices = np.random.choice(self.population_size, self.population_size, replace=False, p=[self.selection_prob, 1 - self.selection_prob])
        selected_indices = selected_indices[:self.population_size // 2]  # Select half of the individuals

        # Evaluate the function for the selected individuals
        selected_evaluations = np.random.rand(len(selected_indices), self.dim)
        selected_evaluations = np.minimum(selected_evaluations, self.search_space)
        selected_evaluations = np.maximum(selected_evaluations, -5.0)
        selected_evaluations = np.minimum(selected_evaluations, 5.0)

        # Perform crossover and mutation
        offspring = []
        for i in range(len(selected_indices)):
            parent1, parent2 = selected_indices[i], selected_indices[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            child1, child2 = self.mutation(child1, child2)
            offspring.append(child1)
            offspring.append(child2)

        # Replace the old population with the new one
        self.population_size = len(offspring)
        self.search_space = np.linspace(-5.0, 5.0, self.dim)
        self.func_evaluations = 0
        for individual in offspring:
            func_value = func(individual)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return offspring

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child1 = parent1[:self.dim // 2] + parent2[self.dim // 2:]
        child2 = parent2[:self.dim // 2] + parent1[self.dim // 2:]
        return child1, child2

    def mutation(self, individual):
        # Perform mutation on an individual
        mutated_individual = individual + np.random.rand(self.dim)
        return mutated_individual

# Example usage:
problem = HSBBO(budget=100, dim=5)
func = lambda x: x**2
individual = problem.__call__(func)
print(individual)