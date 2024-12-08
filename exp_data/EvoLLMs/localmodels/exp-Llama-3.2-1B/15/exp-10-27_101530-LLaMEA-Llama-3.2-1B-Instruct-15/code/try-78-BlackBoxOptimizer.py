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

    def novel_metaheuristic(self, algorithm_name, budget, dim):
        # Initialize the algorithm with the current best solution
        current_solution = self.search_space[0], self.search_space[1]
        current_fitness = self.func(current_solution)

        # Refine the solution using the novel metaheuristic algorithm
        while self.func_evaluations < budget:
            # Evaluate the function at the current solution
            current_fitness = self.func(current_solution)
            # Generate a new solution by refining the current solution
            new_solution = self.refine_solution(current_solution)
            # Evaluate the function at the new solution
            new_fitness = self.func(new_solution)
            # Check if the new solution is better than the current solution
            if new_fitness > current_fitness:
                # If so, update the current solution and fitness
                current_solution = new_solution
                current_fitness = new_fitness

        # Return the best solution found
        return current_solution, current_fitness

    def refine_solution(self, solution):
        # Define the mutation rates
        mutation_rate = 0.1

        # Define the crossover rate
        crossover_rate = 0.5

        # Generate a new solution by mutation
        new_solution = (solution[0] + random.uniform(-2.0, 2.0), solution[1] + random.uniform(-2.0, 2.0))
        # Apply crossover with the current solution
        if random.random() < crossover_rate:
            new_solution = (solution[0] + random.uniform(-2.0, 2.0) * crossover_rate, solution[1] + random.uniform(-2.0, 2.0) * crossover_rate)
        # Apply mutation with the mutation rate
        if random.random() < mutation_rate:
            new_solution = (solution[0] + random.uniform(-2.0, 2.0) * mutation_rate, solution[1] + random.uniform(-2.0, 2.0) * mutation_rate)

        # Evaluate the function at the new solution
        new_fitness = self.func(new_solution)
        # Check if the new solution is better than the current solution
        if new_fitness > current_fitness:
            # If so, update the current solution and fitness
            current_solution = new_solution
            current_fitness = new_fitness

        return new_solution

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 