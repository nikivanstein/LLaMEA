import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        mutation_step = 0.5

        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-mutation_step, mutation_step, self.dim)  # Mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                mutation_step *= 0.95  # Reduce mutation step if fitness improves
            else:
                mutant = np.random.uniform(-5.0, 5.0, self.dim)  # Generate mutant
                trial_solution = best_solution + 0.6 * (mutant - best_solution)  # Create trial solution
                trial_solution = np.clip(trial_solution, -5.0, 5.0)  # Ensure trial solution is within bounds
                trial_fitness = func(trial_solution)
                
                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness
        
        return best_solution