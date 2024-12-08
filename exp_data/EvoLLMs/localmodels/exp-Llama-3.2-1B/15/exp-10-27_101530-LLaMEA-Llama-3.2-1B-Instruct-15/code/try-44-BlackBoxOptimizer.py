import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def novel_metaheuristic(self, func, budget, dim, mutation_rate, exploration_rate):
        # Initialize the population
        population = [func(np.array([random.uniform(self.search_space[0], self.search_space[1]) for _ in range(dim)]) for _ in range(1000))]

        for _ in range(budget):
            # Select the best individual
            best_individual = population[np.argmax([func(individual) for individual in population])]

            # Select a random individual
            individual = population[np.random.randint(0, len(population))]

            # Calculate the fitness of the individual
            fitness = func(individual)

            # If the fitness is better than the best individual, update it
            if fitness > self.search_space[0] + (fitness - self.search_space[0]) * exploration_rate:
                population[np.argmax([func(individual) for individual in population])] = individual

            # If the fitness is worse than the best individual, mutate it
            if fitness < self.search_space[1] - (fitness - self.search_space[1]) * exploration_rate:
                # Randomly select a dimension
                dim_index = random.randint(0, dim - 1)

                # Randomly select a mutation point
                point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))

                # Mutate the individual
                mutated_individual = func(np.array([individual[i] + (point[i] - individual[i]) * mutation_rate for i in range(dim)]))

                # Update the population
                population[np.argmax([func(individual) for individual in population])] = mutated_individual

        # Return the best individual
        return population[np.argmax([func(individual) for individual in population])]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 