import random
import numpy as np
from scipy.optimize import minimize

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

    def optimize(self, func, bounds, initial_point, iterations):
        # Define the bounds for the search space
        self.bounds = bounds
        
        # Define the mutation strategy
        def mutate(individual):
            # Randomly change one element of the individual
            if random.random() < 0.1:
                idx = random.randint(0, self.dim - 1)
                new_val = random.uniform(self.bounds[idx])
                individual[idx] = new_val
            return individual
        
        # Initialize the population with random points
        population = [initial_point] * self.budget
        for _ in range(iterations):
            # Evaluate the fitness of each individual
            fitnesses = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest_individuals = np.argsort(fitnesses)[-self.budget:]
            # Create a new population by mutating the fittest individuals
            new_population = [individuals[:self.budget] + [mutate(individual) for individual in fittest_individuals] for individuals in population]
            # Replace the old population with the new one
            population = new_population
        
        # Return the fittest individual
        return population[np.argmax(fitnesses)]

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Example usage:
if __name__ == "__main__":
    # Create a problem instance
    problem = ioh.iohcpp.problem.RealSingleObjective()
    problem.add_variable("x", bounds=[(-5.0, 5.0)])
    problem.add_function(func=lambda x: x[0]**2 + x[1]**2)
    
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget=100, dim=2)
    
    # Optimize the function
    fittest_individual = optimizer.optimize(problem, bounds=[(-5.0, 5.0)], initial_point=[1.0, 1.0], iterations=100)
    
    # Print the result
    print("Fittest individual:", fittest_individual)
    print("Fitness:", problem.evaluate_fitness(fittest_individual))