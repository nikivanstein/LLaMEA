import numpy as np
from joblib import Parallel, delayed

class Parallel_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            def evaluate_candidate(candidate):
                return func(candidate)
            
            def parallel_evaluate(population):
                return Parallel(n_jobs=-1)(delayed(evaluate_candidate)(candidate) for candidate in population)
            
            # PSO step
            # Update particle positions based on personal and global best
            
            # DE step with hybrid mutation strategy
            for i in range(len(population)):
                candidate = population[i]
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
                    population[i] = trial
        
        population = initialize_population(50)
        while self.budget > 0:
            population_fitness = parallel_evaluate(population)
            optimize_population(population)
            self.budget -= 1
        
        # Return the best solution found
        best_solution = population[np.argmin(population_fitness)]
        return best_solution