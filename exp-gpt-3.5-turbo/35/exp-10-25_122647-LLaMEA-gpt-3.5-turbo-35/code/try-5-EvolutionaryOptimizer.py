import numpy as np

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // self.population_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]
            fitness_values = [func(individual) for individual in population]
            
            for idx, fitness in enumerate(fitness_values):
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = population[idx]
        
        return best_solution