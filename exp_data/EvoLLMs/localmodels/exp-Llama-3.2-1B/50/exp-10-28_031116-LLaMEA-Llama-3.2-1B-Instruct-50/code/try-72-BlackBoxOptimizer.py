import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(100):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            if np.any(solution) == 0:
                solution = np.zeros(self.dim)
            self.population.append(solution)

        # Select the best solution based on the budget
        best_solution = self.population[0]
        best_score = np.inf

        # Run the optimization algorithm for each function
        for func in self.budget:
            # Evaluate the function with the current solution
            score = func(best_solution)

            # If the score is better than the current best score, update the best solution
            if score < best_score:
                best_solution = best_solution
                best_score = score

            # If the budget is exhausted, break the loop
            if len(self.population) == self.budget:
                break

        return best_solution, best_score

    def mutate(self, solution):
        # Randomly change one element in the solution
        idx = random.randint(0, self.dim - 1)
        solution[idx] += random.uniform(-1.0, 1.0)
        return solution

# Define the BBOB test suite of 24 noiseless functions
def bbb(n, dim):
    functions = []
    for i in range(n):
        functions.append(lambda x: x[i] ** 2)
    return functions

# Define the budget and dimensionality
budget = 100
dim = 10

# Initialize the optimizer
optimizer = BlackBoxOptimizer(budget, dim)

# Run the optimization algorithm
best_solution, best_score = optimizer(__call__, bbb)

# Print the results
print("Best Solution:", best_solution)
print("Best Score:", best_score)

# Mutate the best solution to refine its strategy
mutated_solution = optimizer.mutate(best_solution)

# Print the mutated solution
print("Mutated Solution:", mutated_solution)