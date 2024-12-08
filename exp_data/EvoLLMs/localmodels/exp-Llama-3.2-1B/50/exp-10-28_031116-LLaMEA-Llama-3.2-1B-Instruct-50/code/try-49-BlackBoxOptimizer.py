import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Generate a random initial solution within the search space
        initial_solution = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the fitness function
        def fitness(solution):
            return np.linalg.norm(func(solution))

        # Evaluate the fitness function for the initial solution
        fitness_value = fitness(initial_solution)

        # Initialize the population with the initial solution
        self.population.append((initial_solution, fitness_value))

        # Simulate the evolution of the population
        for _ in range(self.budget):
            # Select the next generation based on the probability of refinement
            next_generation = []
            for _ in range(len(self.population)):
                # Refine the solution using the current generation
                if random.random() < 0.45:
                    # Use the current generation to refine the solution
                    next_solution, next_fitness_value = self.refine_solution(self.population[_][0])
                    next_generation.append((next_solution, next_fitness_value))
                else:
                    # Use a simple mutation strategy
                    next_solution = self.mutate_solution(self.population[_][0])
                    next_generation.append((next_solution, self.evaluate_fitness(next_solution)))

            # Replace the old population with the new generation
            self.population = next_generation

    def refine_solution(self, solution):
        # Use a simple heuristic to refine the solution
        refined_solution = solution + np.random.normal(0, 1, self.dim)
        return refined_solution, self.evaluate_fitness(refined_solution)

    def mutate_solution(self, solution):
        # Use a simple mutation strategy to mutate the solution
        mutated_solution = solution + np.random.normal(0, 0.1, self.dim)
        return mutated_solution

    def evaluate_fitness(self, solution):
        # Evaluate the fitness function for the solution
        fitness_value = np.linalg.norm(self.budget * solution)
        return fitness_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a combination of refinement and mutation strategies.