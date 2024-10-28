import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = [[] for _ in range(budget)]
        self.best_solution = None
        self.best_score = -np.inf
        self.search_space = (-5.0, 5.0)

    def __call__(self, func):
        # Evaluate the black box function for the given budget
        for _ in range(self.budget):
            func = func(self.search_space)
            # Select a random point in the search space
            point = random.uniform(self.search_space[0], self.search_space[1])
            # Add the point to the population
            self.population[_].append(point)

        # Select the best solution from the population
        if self.budget == 0:
            return None
        self.best_solution = self.population[0]
        self.best_score = self.population[0][0]

        # Refine the strategy based on the probability of 0.45
        if random.random() < 0.45:
            # Randomly select a point from the population
            point = random.choice(self.population)
            # Evaluate the function at the selected point
            func = func(point)
            # Update the best solution and score if necessary
            if func < self.best_score:
                self.best_solution = point
                self.best_score = func
        else:
            # Use the current best solution
            func = self.best_solution
            # Evaluate the function at the current best solution
            func = func(self.best_solution)
            # Update the best score if necessary
            if func < self.best_score:
                self.best_score = func

        return self.best_solution

# Example usage
optimizer = BlackBoxOptimizer(100, 10)
for name, description, score in [(1, 'Single-objective optimization', 1), (2, 'Multi-objective optimization', 2), (3, 'Non-convex optimization', 3)]:
    func = lambda x: np.sin(x)
    solution = optimizer(func, 10)
    print(f'{name}: {description}, {score}')
    print(f'Solution: {solution}')
    print(f'Score: {optimizer.best_score}')
    print('---')