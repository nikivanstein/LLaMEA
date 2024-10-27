import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('inf')

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
        # Randomly select two points in the search space
        idx1, idx2 = random.sample(range(self.dim), 2)
        # Swap the two points
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        # Ensure the new point is within the search space
        individual[idx1] = np.clip(individual[idx1], self.search_space[0], self.search_space[1])
        individual[idx2] = np.clip(individual[idx2], self.search_space[0], self.search_space[1])
        return individual

    def crossover(self, parent1, parent2):
        # Randomly select a crossover point
        idx = random.randint(1, self.dim - 1)
        # Create a new individual by combining the parents
        child = np.concatenate((parent1[:idx], parent2[idx:]))
        # Return the child individual
        return child

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NMABBO:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def optimize(self, func):
        # Initialize the population with random individuals
        population = [self.optimizer.__call__(func) for _ in range(100)]
        # Evolve the population for a specified number of generations
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitnesses = [individual[1] for individual in population]
            # Select the fittest individuals
            selected_individuals = np.argsort(fitnesses)[-10:]
            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(len(selected_individuals)):
                parent1, parent2 = selected_individuals.pop(0), selected_individuals.pop(0)
                child = self.optimizer.optimize(func)(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            # Replace the old population with the new one
            population = new_population
        # Return the fittest individual in the final population
        return population[0]

# Example usage:
# Create an instance of the NMABBO algorithm
nmabo = NMABBO(1000, 5)

# Optimize the function f(x) = x^2
# Define the function to optimize
def func(x):
    return x**2

# Optimize the function using the NMABBO algorithm
fittest_individual = nmabo.optimize(func)

# Print the fittest individual
print(fittest_individual)