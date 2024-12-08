import random
import numpy as np

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

    def novel_metaheuristic_algorithm(self, func, budget, dim, initial_population, mutation_rate):
        # Initialize the population with random points in the search space
        population = [initial_population]
        for _ in range(budget // 10):  # Reduce the number of iterations for better convergence
            new_population = []
            for _ in range(len(population)):
                # Perform mutation (random walk and linear interpolation)
                mutated_point = self.novel_mutate(population[-1], self.search_space, mutation_rate)
                new_point = self.novel_mutate(mutated_point, self.search_space, mutation_rate)
                new_population.append(new_point)
            population = new_population

        # Select the fittest individual
        fittest_individual = max(population, key=self.evaluate_fitness)

        return fittest_individual

    def novel_mutate(self, individual, search_space, mutation_rate):
        # Randomly select a direction
        direction = np.random.uniform(-1, 1, self.dim)
        # Perform linear interpolation to mutate the individual
        mutated_point = individual + mutation_rate * direction
        # Clip the mutated point to the search space
        mutated_point = np.clip(mutated_point, self.search_space[0], self.search_space[1])
        return mutated_point

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        evaluation = self.func_evaluations
        return evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 