import numpy as np
import random

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        population = [self.evaluate_fitness(func, budget) for _ in range(100)]
        best_individual = population[0]
        best_fitness = population[0]
        for _ in range(100):
            new_individual = self.evaluate_fitness(func, budget)
            if new_individual < best_fitness + 0.05 * (best_fitness - new_individual):
                best_individual = new_individual
                best_fitness = new_individual
            self.search_space = np.vstack((self.search_space, best_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)
        return best_individual

def evaluate_fitness(func, budget):
    return func(self.search_space[np.random.randint(0, self.search_space.shape[0])])

# Initialize the selected solution
selected_solution = BBOBOptimizer(100, 2)

# Evaluate the selected solution
best_individual = selected_solution(__call__(BBOBOptimizer(100, 2)))

# Print the selected solution
print("Best Individual:", best_individual)
print("Best Fitness:", best_individual.__call__(BBOBOptimizer(100, 2)))