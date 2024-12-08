import numpy as np
import random
from scipy.optimize import minimize

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []

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

    def mutate(self, individual):
        perturbation = np.random.uniform(-self.dim, self.dim)
        individual += perturbation
        if np.random.rand() < self.tau:
            perturbation *= self.mu
            individual -= perturbation
        return individual

    def update(self, func, initial_func):
        self.population = [initial_func]
        for _ in range(self.budget):
            self.population.append(func(self.mutate(initial_func)))

        self.best_func = min(self.population, key=lambda x: x.score())

    def evaluate_fitness(self, func, individual):
        return func(individual)

    def score(self, func, individual):
        return func(individual).score

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 