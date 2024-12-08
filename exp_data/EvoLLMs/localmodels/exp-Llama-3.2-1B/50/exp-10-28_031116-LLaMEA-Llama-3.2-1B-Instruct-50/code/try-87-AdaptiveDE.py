import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()
        self.fitness_values = np.zeros((self.population_size, self.dim))

    def generate_initial_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function with the current population
            func_values = differential_evolution(func, self.population, x0=self.population)

            # Refine the population using the adaptive differential evolution strategy
            self.population = self.refine_population(func_values)

            # Evaluate the function with the refined population
            func_values = differential_evolution(func, self.population, x0=self.population)

            # Update the fitness values
            self.fitness_values += func_values.fun

            # Select the next generation based on the fitness values
            self.population = self.select_next_generation(func_values)

    def refine_population(self, func_values):
        # Select the fittest individuals
        fittest_indices = np.argsort(self.fitness_values, axis=1)[:, :self.population_size // 2]

        # Create a new population by combining the fittest individuals with the next generation
        new_population = np.concatenate([self.population[fittest_indices], self.population[:fittest_indices]])

        # Scale the new population to the search space
        new_population = (new_population - np.min(new_population)) / (np.max(new_population) - np.min(new_population))

        return new_population

    def select_next_generation(self, func_values):
        # Select the next generation based on the fitness values
        next_generation = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            next_generation[i] = self.population[i]

        # Refine the next generation using the adaptive differential evolution strategy
        for _ in range(10):
            # Evaluate the function with the current next generation
            func_values = differential_evolution(func, next_generation, x0=self.population)

            # Update the next generation
            next_generation = self.refine_population(func_values)

        return next_generation

    def fitness(self, func_values):
        # Calculate the fitness value for a given function value
        fitness = func_values.fun
        return fitness