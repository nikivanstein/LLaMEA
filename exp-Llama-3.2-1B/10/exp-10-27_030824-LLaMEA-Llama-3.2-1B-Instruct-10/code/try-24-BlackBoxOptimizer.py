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

    def optimize(self, func, max_evaluations=1000):
        # Initialize the population with random individuals
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(50)]

        for _ in range(max_evaluations):
            # Select the fittest individual
            fittest_individual = population[np.argmax([func(individual) for individual in population])]

            # Generate a new individual using linear interpolation
            new_individual = fittest_individual + (np.random.uniform(-1, 1) * (fittest_individual - fittest_individual.mean()))

            # Evaluate the new individual
            new_evaluation = func(new_individual)

            # If the new evaluation is better, replace the fittest individual
            if new_evaluation > fittest_individual:
                population[_] = new_individual

            # If the maximum number of evaluations is reached, stop
            if _ == max_evaluations:
                break

        # Evaluate the best individual
        best_individual, best_evaluation = population[0], population[0]
        for individual, evaluation in zip(population, [func(individual) for individual in population]):
            if evaluation > best_evaluation:
                best_individual, best_evaluation = individual, evaluation

        return best_individual, best_evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.