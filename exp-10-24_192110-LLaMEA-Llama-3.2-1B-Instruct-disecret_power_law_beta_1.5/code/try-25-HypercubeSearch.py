import numpy as np
from scipy.optimize import minimize
import random

class HypercubeSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x0 = np.random.uniform(-5.0, 5.0, dim)
        self.fitness = -np.inf
        self.population = [[self.x0]]  # Initialize the population with a single solution

    def __call__(self, func):
        while self.fitness < self.budget:
            # Generate a new solution using the current population
            x = [self.population[i][0] for i in range(self.population.index(self.x0) + 1)]
            x.append(self.x0)

            # Evaluate the function at the new solution
            res = minimize(func, x, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)

            # Update the fitness and the population
            self.fitness = -res.fun
            self.population.append(x)

    def select_solution(self):
        # Select the fittest solution based on the fitness
        selected_solution = max(self.population, key=self.fitness)
        return selected_solution

    def adapt(self, mutation_rate):
        # Randomly select a mutation point in the population
        mutation_point = random.randint(0, len(self.population) - 1)

        # Randomly swap two points in the population
        i, j = mutation_point, random.randint(0, len(self.population) - 1)
        self.population[i], self.population[j] = self.population[j], self.population[i]

# Select the initial solution
budget = 1000
dim = 5
x0 = np.random.uniform(-5.0, 5.0, dim)
search = HypercubeSearch(budget, dim)

# Evaluate the function at the initial solution
x0 = search.x0
res = search(func, x0)

# Select the initial solution
solution = search.select_solution()

# Initialize the population
search.population = [[x0]]  # Initialize the population with a single solution

# Run the evolutionary optimization algorithm
for _ in range(100):
    # Select the fittest solution
    selected_solution = search.select_solution()

    # Adapt the solution
    mutation_rate = 0.01
    if random.random() < mutation_rate:
        # Randomly swap two points in the population
        i, j = random.randint(0, len(search.population) - 1), random.randint(0, len(search.population) - 1)
        search.population[i], search.population[j] = search.population[j], search.population[i]

# Evaluate the function at the final solution
x0 = search.x0
res = search.func(x0)

# Print the results
print("Initial Solution:", selected_solution)
print("Final Solution:", x0)
print("Fitness:", -res.fun)
print("Population:", search.population)