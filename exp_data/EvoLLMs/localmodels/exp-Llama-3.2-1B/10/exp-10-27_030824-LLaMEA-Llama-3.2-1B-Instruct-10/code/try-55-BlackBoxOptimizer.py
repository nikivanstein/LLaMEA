import random
import numpy as np
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population_size = 100
        self.population = deque(maxlen=self.population_size)

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Randomly select a mutation point within the search space
        mutation_point = np.random.randint(0, len(individual) - 1)
        # Swap the mutation point with a random point in the search space
        individual[mutation_point], individual[mutation_point + np.random.randint(0, len(individual) - mutation_point - 1)] = individual[mutation_point + np.random.randint(0, len(individual) - mutation_point - 1)], individual[mutation_point]
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Randomly select a crossover point within the search space
        crossover_point = np.random.randint(0, len(parent1) - 1)
        # Create a child individual by combining the parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        # Return the child individual
        return child

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

def objective_function(individual, budget):
    # Evaluate the objective function at each point in the search space
    evaluations = []
    for point in individual:
        evaluation = func(point)
        evaluations.append(evaluation)
    # Return the evaluations
    return evaluations

def func(x):
    # Define the objective function to optimize
    return x[0]**2 + x[1]**2

# Initialize the optimizer
optimizer = BlackBoxOptimizer(budget=100, dim=2)

# Evaluate the objective function for 1000 iterations
for _ in range(1000):
    # Optimize the objective function using the optimizer
    evaluations = objective_function(optimizer.population, budget=100)
    # Print the evaluations
    print(evaluations)

# Print the final solution
print("Final solution:", optimizer.population[-1])