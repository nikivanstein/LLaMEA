import numpy as np
import random
import copy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the population
        self.population = [copy.deepcopy(func) for _ in range(100)]

        # Try different initializations
        for _ in range(10):
            # Randomly initialize the population
            self.population = [random.uniform(bounds, size=self.dim) for _ in range(100)]

            # Evaluate the function at the population
            fitness = [self.__call__(func, individual) for individual in self.population]

            # Select the fittest individuals
            self.population = sorted(self.population, key=lambda individual: fitness[individual], reverse=True)[:self.budget]

            # Refine the population using probability 0.25
            self.population = [individual if random.random() < 0.25 else random.choice(self.population) for individual in self.population]

        # Return the best solution found
        return self.population[0]

# One-line description with main idea
# "Evolutionary Algorithm for Black Box Optimization using Genetic Programming"
# that uses a population-based approach with mutation and selection to optimize black box functions"