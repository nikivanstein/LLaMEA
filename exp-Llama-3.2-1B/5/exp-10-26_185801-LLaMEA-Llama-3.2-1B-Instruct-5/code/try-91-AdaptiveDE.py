# Description: Adaptive Differential Evolution Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class AdaptiveDE:
    def __init__(self, budget, dim, mutation_rate, crossover_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Initialize population with random individuals
        population = self.generate_population(self.budget)

        # Evaluate population fitness
        fitness_values = self.evaluate_fitness(population, func)

        # Select fittest individuals for mutation
        fittest_individuals = self.select_fittest(population, fitness_values)

        # Perform mutation on fittest individuals
        mutated_individuals = self.mutate(fittest_individuals, fitness_values)

        # Evaluate new population fitness
        new_fitness_values = self.evaluate_fitness(mutated_individuals, func)

        # Select new fittest individuals
        new_fittest_individuals = self.select_fittest(mutated_individuals, new_fitness_values)

        # Replace old population with new one
        population = new_fittest_individuals

        # Update fitness values
        self.func_evaluations += 1
        self.search_space = np.linspace(-5.0, 5.0, self.dim)

        # Return selected individual
        return self.select_fittest(population, new_fitness_values)[0]

    def generate_population(self, budget):
        population = []
        for _ in range(budget):
            individual = np.random.uniform(self.search_space)
            population.append(individual)
        return population

    def evaluate_fitness(self, population, func):
        fitness_values = []
        for individual in population:
            func_value = func(individual)
            fitness_values.append(func_value)
        return fitness_values

    def select_fittest(self, population, fitness_values):
        fittest_individuals = population[np.argmax(fitness_values)]
        return fittest_individuals

    def mutate(self, individuals, fitness_values):
        mutated_individuals = []
        for individual in individuals:
            mutated_individual = individual.copy()
            if np.random.rand() < self.mutation_rate:
                mutated_individual = self.mutate_individual(mutated_individual, fitness_values)
            mutated_individuals.append(mutated_individual)
        return mutated_individuals

    def mutate_individual(self, individual, fitness_values):
        mutation_rate = self.mutation_rate
        for i in range(self.dim):
            mutation = random.uniform(-5.0, 5.0)
            individual[i] += mutation
            if individual[i] < -5.0:
                individual[i] = -5.0
            elif individual[i] > 5.0:
                individual[i] = 5.0
            if np.isnan(individual[i]) or np.isinf(individual[i]):
                raise ValueError("Invalid function value")
            if individual[i] < 0 or individual[i] > 1:
                raise ValueError("Function value must be between 0 and 1")
        return individual

    def evaluateBBOB(self, func):
        # Evaluate black box function
        func_value = func(self.search_space)
        return func_value