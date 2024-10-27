import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.metaheuristic_iterations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Refine the strategy using metaheuristic iterations
            self.metaheuristic_iterations += 1
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

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.metaheuristic_iterations = 0

    def __call__(self, func):
        # Define the initial population
        population = [BlackBoxOptimizer(self.budget, self.dim) for _ in range(100)]
        # Define the mutation function
        def mutate(individual):
            # Refine the strategy using metaheuristic iterations
            self.metaheuristic_iterations += 1
            # Generate a new point using the refined strategy
            new_point = (individual.search_space[0] + random.uniform(-1, 1), 
                        individual.search_space[1] + random.uniform(-1, 1))
            # Evaluate the function at the new point
            new_func_value = func(new_point)
            # Check if the new point is within the budget
            if new_func_value <= 100:
                # If not, mutate the individual
                return mutate(individual)
            else:
                # If the new point is within the budget, return the individual
                return individual
        # Evaluate the fitness of each individual in the population
        for individual in population:
            individual.evaluate_fitness(func)
        # Select the fittest individuals
        selected_individuals = sorted(population, key=lambda individual: individual.evaluate_fitness(func), reverse=True)[:self.budget]
        # Return the fittest individual
        return selected_individuals[0]

# Example usage:
func = lambda x: np.sin(x)
optimizer = NovelMetaheuristicOptimizer(100, 10)
best_individual = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_individual.evaluate_fitness(func))