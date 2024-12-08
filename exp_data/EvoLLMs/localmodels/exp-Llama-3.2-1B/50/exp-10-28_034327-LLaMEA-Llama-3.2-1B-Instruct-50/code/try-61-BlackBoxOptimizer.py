import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        def evaluate_fitness(individual):
            new_individual = individual
            for _ in range(self.budget):
                # Refine strategy by changing individual lines
                new_individual = self.iterated_permutation_and_cooling(individual)
                # Evaluate the function at the new individual
                value = func(new_individual)
                # Check if the new individual is within the bounds
                if -5.0 <= new_individual[0] <= 5.0 and -5.0 <= new_individual[1] <= 5.0:
                    # If the new individual is within bounds, update the function value
                    self.func_evals += 1
                    return value
            # If the budget is exceeded, return the best point found so far
            return np.max(func(new_individual))

        return evaluate_fitness

    def iterated_permutation_and_cooling(self, individual):
        # Generate a random permutation of the individual's coordinates
        perm = list(individual)
        for _ in range(self.dim):
            perm.append(random.choice([0, 1]))
        # Shuffle the permutation
        random.shuffle(perm)
        # Update the individual with the new permutation
        individual = tuple(perm)
        return individual

# Example usage
budget = 1000
dim = 2
optimizer = BlackBoxOptimizer(budget, dim)

# Evaluate the function
func = lambda x: x[0]**2 + x[1]**2
best_individual = optimizer(func)
print("Best individual:", best_individual)
print("Best function value:", optimizer(best_individual))