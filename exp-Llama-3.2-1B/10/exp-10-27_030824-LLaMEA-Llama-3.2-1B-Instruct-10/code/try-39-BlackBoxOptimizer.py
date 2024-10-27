import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

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

    def optimize(self, func, max_iter=1000, tol=1e-6):
        # Initialize the population with random points in the search space
        population = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Run the evolution algorithm
        for _ in range(max_iter):
            # Evaluate the fitness of each individual in the population
            fitness = [func(individual) for individual in population]
            # Calculate the differences between the fitness values
            diffs = [fitness[i] - fitness[i+1] for i in range(len(fitness)-1)]
            # Sort the differences in descending order
            diffs.sort(reverse=True)

            # Select the fittest individuals to reproduce
            selected_individuals = population[diffs.index(max(diffs))[:50]] + population[diffs.index(max(diffs))[-50:]]
            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(100):
                parent1, parent2 = random.sample(selected_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.5:
                    # Perform mutation by adding a random noise to the child
                    child += np.random.uniform(-1, 1)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        # Return the fittest individual in the final population
        return population[0]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.