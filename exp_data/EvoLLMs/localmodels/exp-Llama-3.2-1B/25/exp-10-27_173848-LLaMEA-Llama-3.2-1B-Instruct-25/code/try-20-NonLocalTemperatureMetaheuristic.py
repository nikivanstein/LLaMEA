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
        self.population = None

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

    def initialize(self, func, dim):
        self.population = [func(np.random.uniform(-dim, dim)) for _ in range(100)]

    def fitness(self, individual):
        return np.mean([self.evaluate_fitness(individual) for individual in self.population])

    def mutate(self, individual):
        # Refine the strategy
        perturbation = np.random.uniform(-self.dim, self.dim)
        updated_individual = individual + perturbation
        updated_individual = np.clip(updated_individual, -self.dim, self.dim)
        return updated_individual

    def evaluate_fitness(self, individual):
        new_func = self.population[0] + individual + random.uniform(-self.dim, self.dim)
        return np.mean([self.evaluate_fitness(new_func) for _ in range(self.budget)])

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 