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

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

def mutate(individual):
    # Select two random points in the search space
    point1 = individual[:self.dim // 2]
    point2 = individual[self.dim // 2:]

    # Swap the two points
    individual = np.concatenate((point1, point2))

    return individual

def selection(population, budget):
    # Select the fittest individuals
    fittest = sorted(population, key=lambda individual: individual[func_evaluations], reverse=True)[:budget]

    # Select two individuals
    individual1 = fittest[0]
    individual2 = fittest[1]

    # Select a random point in the search space
    point = np.random.choice(self.search_space)

    # Swap the two individuals
    individual1 = mutate(individual1)
    individual2 = mutate(individual2)

    # Merge the two individuals
    individual1 = np.concatenate((individual1, point))

    return individual1, individual2

def crossover(parent1, parent2):
    # Select two random points in the search space
    point1 = parent1[:self.dim // 2]
    point2 = parent2[self.dim // 2:]

    # Swap the two points
    parent1 = np.concatenate((point1, point2))

    return parent1

def __call__(self, func):
    population = [func(x) for x in np.random.uniform(self.search_space, size=self.dim)]

    while len(population) < self.budget:
        # Select two individuals
        individual1, individual2 = selection(population, self.budget - len(population))

        # Crossover the two individuals
        child = crossover(individual1, individual2)

        # Mutate the child
        child = mutate(child)

        # Add the child to the population
        population.append(child)

    # Return the fittest individual
    return max(population, key=func)