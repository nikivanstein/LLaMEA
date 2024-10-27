import numpy as np
import random
import copy

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
        if len(self.population) < self.budget:
            self.population.append(copy.deepcopy(individual))
        else:
            self.population = self.population[:self.budget]

        # Refine the strategy
        perturbation = np.random.uniform(-self.dim, self.dim)
        if random.random() < 0.25:
            perturbation *= self.mu
        new_individual = individual + perturbation

        # Check if the new individual is better
        if np.random.rand() < self.alpha:
            new_individual = copy.deepcopy(new_individual)
        else:
            new_individual *= self.tau

        # Update the population
        self.population = self.population[:self.budget]

        return new_individual

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 