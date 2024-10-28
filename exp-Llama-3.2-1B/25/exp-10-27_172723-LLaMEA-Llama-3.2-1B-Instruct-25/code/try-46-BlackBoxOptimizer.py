import random
import numpy as np
from scipy.optimize import differential_evolution

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
        idx1, idx2 = random.sample(range(self.dim), 2)
        self.search_space[idx1] += random.uniform(-1, 1)
        self.search_space[idx2] += random.uniform(-1, 1)
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point and create a new individual by combining the two parents
        crossover_point = random.randint(1, self.dim - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

# Example usage:
optimizer = BlackBoxOptimizer(100, 5)
problem = RealSingleObjectiveProblem(1, "Sphere", 1.0, 5.0)
best_solution = optimizer(problem, 10)
print("Best solution:", best_solution)