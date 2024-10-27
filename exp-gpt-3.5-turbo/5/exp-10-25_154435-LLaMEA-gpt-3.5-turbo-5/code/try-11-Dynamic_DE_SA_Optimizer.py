import numpy as np

class Dynamic_DE_SA_Optimizer:
    def __init__(self, budget, dim, population_size=50, crossover_prob=0.9, cooling_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.cooling_rate = cooling_rate

    def dynamic_mutation_factor(self, iteration, max_iterations, initial_factor=0.8, final_factor=0.2):
        return initial_factor + (final_factor - initial_factor) * iteration / max_iterations

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])
        
        for iteration in range(self.budget):
            mutation_factor = self.dynamic_mutation_factor(iteration, self.budget)
            # Remaining code same as DE_SA_Optimizer for the main loop
            
        return best_solution