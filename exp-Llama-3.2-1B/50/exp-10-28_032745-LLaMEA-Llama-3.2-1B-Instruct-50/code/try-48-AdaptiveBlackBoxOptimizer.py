import numpy as np
from scipy.optimize import minimize
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population_size = 100
        self.population_mutation_rate = 0.01
        self.population_crossover_rate = 0.5
        self.population_selection_rate = 0.1

    def __call__(self, func, initial_guess, iterations):
        population = self.generate_population(self.population_size, self.dim)
        best_individual = population[0]
        best_value = self.func(best_individual)
        for _ in range(iterations):
            if _ >= self.budget:
                break
            for i in range(self.population_size):
                fitness = self.evaluate_fitness(population[i])
                if fitness < best_value:
                    best_individual = population[i]
                    best_value = fitness
            # Select the best individual based on the selection rate
            selected_indices = np.random.choice(self.population_size, size=self.population_size, replace=False, p=[1 - self.population_selection_rate, self.population_selection_rate])
            selected_individuals = [population[i] for i in selected_indices]
            # Perform crossover and mutation
            for i in range(self.population_size):
                if random.random() < self.population_crossover_rate:
                    parent1, parent2 = random.sample(selected_individuals, 2)
                    child = (parent1 + parent2) / 2
                    if random.random() < self.population_mutation_rate:
                        child[0] += random.uniform(-0.01, 0.01)
                        child[1] += random.uniform(-0.01, 0.01)
            population = selected_individuals + [child]
        return best_individual, best_value

    def generate_population(self, population_size, dim):
        return [np.random.uniform(self.search_space[0], self.search_space[1], (dim,)) for _ in range(population_size)]

    def evaluate_fitness(self, individual):
        return self.func(individual)

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using adaptive search strategy
# 
# The algorithm uses a population-based approach with adaptive search strategy, where the strategy is adjusted based on the performance of the individuals in the population.