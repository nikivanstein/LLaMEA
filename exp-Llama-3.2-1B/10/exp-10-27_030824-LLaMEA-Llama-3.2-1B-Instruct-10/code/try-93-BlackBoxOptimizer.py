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

    def update_individual(self, individual, new_point, new_evaluation):
        # Calculate the probability of selecting the new point
        probabilities = np.random.rand(len(individual))
        probabilities /= probabilities.sum()
        # Select the new point with the highest probability
        selected_point = np.random.choice(len(individual), p=probabilities)
        # Update the individual with the new point
        new_individual = individual[:selected_point] + [new_point] + individual[selected_point:]
        # Evaluate the new individual
        new_evaluation = func(new_individual)
        # Return the new individual and its evaluation
        return new_individual, new_evaluation

    def mutation(self, individual):
        # Select a random point in the search space
        point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Update the individual with the new point
        individual = [point] + individual[:-1] + [point]
        # Evaluate the new individual
        evaluation = func(individual)
        # Return the new individual and its evaluation
        return individual, evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def optimize_function(self, func):
        # Initialize the population with random individuals
        population = [np.random.uniform(self.optimizer.search_space[0], self.optimizer.search_space[1]) for _ in range(50)]
        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.optimizer(individual, func) for individual in population]
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)[-10:]]
            # Create a new population by combining the fittest individuals with new points
            new_population = []
            for _ in range(len(fittest_individuals)):
                # Select a random point in the search space
                point = np.random.uniform(self.optimizer.search_space[0], self.optimizer.search_space[1])
                # Combine the fittest individual with the new point
                new_individual = fittest_individuals[_] + [point]
                # Evaluate the new individual
                new_evaluation = func(new_individual)
                # Add the new individual to the new population
                new_population.append(new_individual)
            # Replace the old population with the new population
            population = new_population
        # Return the optimized function
        return func, population

# Usage
optimizer = NovelMetaheuristicOptimizer(budget=100, dim=5)
func, population = optimizer.optimize_function(lambda x: x**2)
print("Optimized function:", func)
print("Optimized population:", population)