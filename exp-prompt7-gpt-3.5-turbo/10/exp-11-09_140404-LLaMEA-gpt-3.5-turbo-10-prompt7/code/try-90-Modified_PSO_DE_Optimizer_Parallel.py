import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Modified_PSO_DE_Optimizer_Parallel:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            with ThreadPoolExecutor() as executor:
                # PSO step
                # Update particle positions based on personal and global best
                executor.map(particle_optimization, population)
        
        def particle_optimization(candidate):
            # DE step with hybrid mutation strategy
            # Mutation strategy 1
            mutant1 = population[np.random.choice(len(population))]
            # Mutation strategy 2
            mutant2 = candidate + 0.5 * (population[np.random.choice(len(population))] - candidate)
            
            # Additional mutation step with dynamically adjusted mutation rate
            mutation_rate = 0.5 / np.sqrt(np.sqrt(self.budget))  # Dynamic mutation rate
            mutant3 = candidate + mutation_rate * np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Selection strategy
            trial = mutant1 if func(mutant1) < func(mutant2) else mutant2
            trial = mutant3 if func(mutant3) < func(trial) else trial
            
            if func(trial) < func(candidate):
                candidate = trial
        
        population = initialize_population(50)
        while self.budget > 0:
            optimize_population(population)
            self.budget -= 1
        
        # Return the best solution found
        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution