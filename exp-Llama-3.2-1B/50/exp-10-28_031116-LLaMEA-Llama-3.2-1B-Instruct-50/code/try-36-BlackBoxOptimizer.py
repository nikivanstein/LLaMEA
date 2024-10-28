import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.selection_prob = 0.45
        self.mutation_prob = 0.01

    def initialize_population(self):
        # Initialize population with random solutions
        solutions = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            solutions.append(solution)
        return np.array(solutions)

    def __call__(self, func):
        # Evaluate the function using self.budget function evaluations
        func_evaluations = np.zeros(self.budget)
        for i, func in enumerate(func):
            func_evaluations[i] = func(self.population[i])
        
        # Select the best solutions
        selected_solutions = self.select_solutions(func_evaluations, self.budget)
        
        # Mutate the selected solutions
        mutated_solutions = self.mutate(selected_solutions, func_evaluations)
        
        # Evaluate the mutated solutions
        mutated_func_evaluations = np.zeros(len(mutated_solutions))
        for i, func in enumerate(mutated_solutions):
            func_evaluations[i] = func(self.population[i])
        
        # Select the best mutated solutions
        best_solutions = self.select_solutions(mutated_func_evaluations, self.budget)
        
        # Replace the old population with the new population
        self.population = np.array(best_solutions)
        
        # Update the fitness scores
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        for i, func in enumerate(best_solutions):
            func_evaluations[i] = func(self.population[i])
            self.fitness_scores[i] = func_evaluations[i]
        
        # Return the best solution
        return best_solutions[np.argmax(self.fitness_scores)]

    def select_solutions(self, func_evaluations, budget):
        # Select the solutions with the highest fitness scores
        selected_solutions = np.argsort(-func_evaluations)
        return selected_solutions[:budget]

    def mutate(self, solutions, func_evaluations):
        # Randomly mutate the solutions
        mutated_solutions = solutions.copy()
        for _ in range(self.population_size):
            idx = random.randint(0, self.population_size - 1)
            mutated_solutions[idx] = func_evaluations[idx] + random.uniform(-1, 1) / 10.0
        return mutated_solutions

# Description: Black Box Optimization using Genetic Algorithm with Adaptive Mutation and Selection
# Code: 