import random
import numpy as np
from scipy.optimize import minimize

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
        # Randomly change a single element in the individual
        index = random.randint(0, self.dim - 1)
        new_individual = individual.copy()
        new_individual[index] = random.uniform(-5.0, 5.0)
        return new_individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

class BBOB:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)
        self.population_size = 100
        self.population = [self.optimizer.__call__(lambda x: x) for _ in range(self.population_size)]

    def select(self):
        # Select the fittest individual from the population
        return self.population[np.argmax([self.optimizer.__call__(individual) for individual in self.population])]

    def mutate(self, individual):
        # Mutate the selected individual
        return self.optimizer.mutate(individual)

    def evolve(self):
        # Evolve the population using mutation and selection
        new_population = [self.optimizer.__call__(individual) for individual in self.population]
        self.population = new_population

# Usage:
bbo = BBOB(budget=100, dim=5)
bbo.evolve()
print(bbo.select())
print(bbo.mutate(bbo.select()))