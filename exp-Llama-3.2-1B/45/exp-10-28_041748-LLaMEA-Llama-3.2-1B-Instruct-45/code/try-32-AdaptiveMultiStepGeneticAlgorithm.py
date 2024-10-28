# Description: Adaptive Multi-Step Genetic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from collections import deque

class AdaptiveMultiStepGeneticAlgorithm:
    def __init__(self, budget, dim, population_size, crossover_rate, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = deque(maxlen=1000)

    def __call__(self, func):
        # Initialize the population
        for _ in range(self.population_size):
            self.population.append(func(np.random.uniform(self.search_space[0], self.search_space[1], dim)))

        # Evaluate the function with the given budget
        func_evaluations = np.array([np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.budget)])

        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        top_individuals = np.array([self.population[i] for i in top_individuals])

        # Create a new population by crossover and mutation
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)

        # Replace the old population with the new one
        self.population = deque(new_population)

        # Evaluate the new population
        new_func_evaluations = np.array([np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(len(new_population))])
        new_func_evaluations = np.array([func(x) for x in new_func_evaluations])

        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

# One-line description with the main idea
# Adaptive Multi-Step Genetic Algorithm for Black Box Optimization