import numpy as np

class Accelerated_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            # Pure PSO step
            # Update particle positions based on personal and global best
            for i in range(len(population)):
                candidate = population[i]
                # PSO update rule
                new_position = candidate + np.random.uniform() * (population[np.argmin([func(p) for p in population])] - candidate)
                
                if func(new_position) < func(candidate):
                    population[i] = new_position
        
        population = initialize_population(50)
        while self.budget > 0:
            optimize_population(population)
            self.budget -= 1
        
        # Return the best solution found
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution