import numpy as np

class EnhancedMetaheuristicAlgorithmRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_scale = 0.5  # Initialize mutation scale

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for eval_count in range(self.budget):
            if eval_count % (self.budget // 10) == 0 and eval_count > 0:
                self.mutation_scale = 0.5 - 0.5 * eval_count / self.budget  # Adapt mutation scale

            candidate_solution = 0.8 * best_solution + 0.2 * np.random.uniform(-5.0, 5.0, self.dim)  # Modified mutation step with weighted average
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            else:
                mutant = np.random.uniform(-5.0, 5.0, self.dim)  # Generate mutant
                trial_solution = best_solution + 0.6 * (mutant - best_solution)  # Create trial solution
                trial_solution = np.clip(trial_solution, -5.0, 5.0)  # Ensure trial solution is within bounds
                trial_fitness = func(trial_solution)
                
                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness
        
        return best_solution