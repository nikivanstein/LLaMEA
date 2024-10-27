import numpy as np
import random
import math

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.mutation_rate = 0.1

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

            # Calculate the non-local temperature
            non_local_temp = math.exp((self.temp - self.alpha * self.mu) / 100)

            # Apply mutation
            if random.random() < self.mutation_rate:
                # Randomly select an individual from the search space
                individual = np.random.choice(self.dim, size=self.dim, replace=False)

                # Update the individual using the non-local temperature
                updated_individual = self.evaluate_fitness(individual + perturbation, non_local_temp)

                # Revert the perturbation
                perturbation *= self.tau
                updated_individual = individual + perturbation

                # Check if the updated individual is better
                if np.random.rand() < self.alpha:
                    updated_func = self.evaluate_fitness(updated_individual, non_local_temp)
                else:
                    updated_func = func + perturbation

                # Update the best function
                self.best_func = updated_func

            num_evals += 1

        return self.best_func

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Mutation
# Code: 