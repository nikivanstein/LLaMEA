import numpy as np
import random

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            num_evals += 1

        return self.best_func

class EvolutionaryOptimization:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []

    def __call__(self, func, population_size=100):
        for _ in range(population_size):
            # Select a random individual from the population
            individual = np.random.uniform(-self.dim, self.dim, self.dim)

            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual, func)

            # Add the individual to the population
            self.population.append((individual, fitness))

        # Select the best individual
        self.best_func = self.population[0][1]

        # Refine the strategy
        for _ in range(self.budget):
            # Select a random individual from the population
            individual = np.random.choice(self.population, self.dim, replace=False)

            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual, func)

            # Update the individual
            self.population[self.population.index((individual, fitness))][0] = individual

            # Check if the new individual is better
            if np.random.rand() < self.alpha:
                self.population[self.population.index((individual, fitness))][1] = fitness

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 