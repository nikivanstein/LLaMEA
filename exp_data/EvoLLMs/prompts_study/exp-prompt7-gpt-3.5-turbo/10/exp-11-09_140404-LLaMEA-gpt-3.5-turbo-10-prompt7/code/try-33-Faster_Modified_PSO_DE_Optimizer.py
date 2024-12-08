import numpy as np
import concurrent.futures

class Faster_Modified_PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(-5.0, 5.0, size=(size, self.dim))
        
        def optimize_population(population):
            def evaluate_candidate(candidate):
                return func(candidate)
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                fitness_results = list(executor.map(evaluate_candidate, population))
            
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
                trial = mutant1 if fitness_results[np.where(population == mutant1)[0][0]] < fitness_results[np.where(population == mutant2)[0][0]] else mutant2
                trial = mutant3 if fitness_results[np.where(population == mutant3)[0][0]] < fitness_results[np.where(population == trial)[0][0]] else trial
                
                if fitness_results[i] < func(trial):
                    population[i] = trial
                    
        population = initialize_population(50)
        while self.budget > 0:
            optimize_population(population)
            self.budget -= 1
        
        # Return the best solution found
        best_solution = population[np.argmin(fitness_results)]
        return best_solution