import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def optimize(self, func):
        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Define the bounds for the search space
        upper = 1e2
        
        # Run differential evolution to find the optimal solution
        result = differential_evolution(func, bounds, args=(upper,))
        
        # Refine the solution based on the probability 0.45
        refined_solution = self._refine_solution(result.x, 0.45)
        
        # Return the refined solution
        return refined_solution

    def _refine_solution(self, solution, probability):
        # Refine the solution based on the probability
        if random.random() < probability:
            # Select a new solution with a higher fitness
            new_solution = solution
        else:
            # Select a new solution with a lower fitness
            new_solution = solution - 0.1
        
        # Return the refined solution
        return new_solution

# Description: Differential Evolution
# Code: 