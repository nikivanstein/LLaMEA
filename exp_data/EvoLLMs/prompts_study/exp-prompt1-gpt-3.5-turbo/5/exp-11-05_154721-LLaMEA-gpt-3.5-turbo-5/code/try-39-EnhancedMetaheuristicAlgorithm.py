import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.5

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-0.5, 0.5, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                self.mutation_rate = max(0.1, self.mutation_rate * 0.9)  # Adjust mutation rate for successful improvements
            else:
                mutant = np.random.uniform(-5.0, 5.0, self.dim)
                trial_solution = best_solution + self.mutation_rate * (mutant - best_solution)
                trial_solution = np.clip(trial_solution, -5.0, 5.0)
                trial_fitness = func(trial_solution)
                
                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness
                    self.mutation_rate = min(0.9, self.mutation_rate * 1.1)  # Increase mutation rate for exploring new regions
        
        return best_solution