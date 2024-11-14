import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Parallel_Modified_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            # PSO step
            # Update particle positions based on personal and global best
            
            # DE step with hybrid mutation strategy
            def optimize_individual(candidate):
                mutant1 = population[np.random.choice(len(population))]
                mutant2 = candidate + 0.5 * (population[np.random.choice(len(population))] - candidate)
                mutation_rate = 0.5 / np.sqrt(np.sqrt(self.budget))
                mutant3 = candidate + mutation_rate * np.random.uniform(-5.0, 5.0, size=self.dim)
                trial = mutant1 if func(mutant1) < func(mutant2) else mutant2
                trial = mutant3 if func(mutant3) < func(trial) else trial
                return trial if func(trial) < func(candidate) else candidate

            with ThreadPoolExecutor() as executor:
                population = list(executor.map(optimize_individual, population))

        population = initialize_population(50)
        while self.budget > 0:
            optimize_population(population)
            self.budget -= 1
        
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution