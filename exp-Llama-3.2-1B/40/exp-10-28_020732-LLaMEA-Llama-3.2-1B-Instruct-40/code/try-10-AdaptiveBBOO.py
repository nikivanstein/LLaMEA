import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import differential_evolution

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.population = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        # Update the population
        self.population.append((x, func_value))

        # Refine the strategy
        if len(self.population) > self.budget * 0.4:
            # Select the best individual based on the fitness
            best_individual, best_fitness = self.population[0]

            # Select the next individual based on the probability of refinement
            if random.random() < 0.4:
                # Select the next individual from the best individual
                next_individual = best_individual
            else:
                # Select the next individual from the population
                next_individual = random.choice(self.population)

            # Refine the next individual
            next_individual = self.refine(next_individual)

            # Update the population
            self.population.append((next_individual, best_fitness))

    def refine(self, individual):
        # Define the mutation function
        def mutate(individual):
            mutated_individual = individual.copy()
            for i in range(self.dim):
                if random.random() < 0.1:
                    # Randomly select a random point in the search space
                    idx = random.randint(0, self.search_space.shape[0] - 1)

                    # Perturb the point
                    mutated_individual[idx] += random.uniform(-1, 1)

            return mutated_individual

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select the parent with the higher fitness
            if random.random() < 0.5:
                parent = parent1
            else:
                parent = parent2

            # Select the crossover point
            crossover_point = random.randint(0, self.search_space.shape[0] - 1)

            # Split the parent into two segments
            segment1 = parent[:crossover_point]
            segment2 = parent[crossover_point:]

            # Combine the two segments
            child = segment1 + segment2

            return child

        # Perform the mutation
        mutated_individual = mutate(individual)

        # Perform the crossover
        child = crossover(mutated_individual, individual)

        return child

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function
bboo.optimize(func)

# Print the results
print(f"Best solution: {bboo.population[0][0]}, Best function value: {bboo.population[0][1]}")