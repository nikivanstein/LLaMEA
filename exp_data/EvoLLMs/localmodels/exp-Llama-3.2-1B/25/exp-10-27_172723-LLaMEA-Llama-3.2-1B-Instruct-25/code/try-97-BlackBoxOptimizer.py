import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def mutate(self, individual):
        # Randomly change two elements in the individual
        i, j = random.sample(range(self.dim), 2)
        self.search_space[i], self.search_space[j] = self.search_space[j], self.search_space[i]
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover on two parents to create a child
        child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
        return child

    def selection(self, population):
        # Select the fittest individuals in the population
        return sorted(population, key=self.func_evaluations, reverse=True)[:self.budget]

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"