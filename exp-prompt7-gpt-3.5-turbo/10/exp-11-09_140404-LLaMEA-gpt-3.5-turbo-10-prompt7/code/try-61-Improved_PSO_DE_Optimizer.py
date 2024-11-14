import numpy as np
from concurrent.futures import ProcessPoolExecutor

class Improved_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            def evaluate(candidate):
                return func(candidate)
            
            # PSO and DE steps remain unchanged
            
            # Parallel evaluation of candidate solutions
            with ProcessPoolExecutor() as executor:
                fitness_results = list(executor.map(evaluate, population))
            
            for i in range(len(population)):
                # Selection strategy with updated fitness results
                # Unchanged code for selection and update steps
            
        population = initialize_population(50)
        while self.budget > 0:
            optimize_population(population)
            self.budget -= 1
        
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution