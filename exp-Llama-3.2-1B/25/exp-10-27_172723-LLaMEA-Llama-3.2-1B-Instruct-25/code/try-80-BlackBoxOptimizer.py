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
        # Randomly change two random genes in the individual
        if random.random() < 0.25:
            # Change two random genes
            individual = np.random.choice(self.search_space, size=self.dim, replace=False)
            # Update the genes
            individual[0], individual[1] = individual[1], individual[0]
        return individual

    def crossover(self, individual1, individual2):
        # Select a random crossover point
        crossover_point = random.randint(1, self.dim - 1)
        # Split the two individuals
        child1 = individual1[:crossover_point]
        child2 = individual2[crossover_point:]
        # Combine the two children
        child = np.concatenate((child1, child2))
        return child

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"