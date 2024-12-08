import numpy as np

class DynamicMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  
        best_fitness = func(best_solution)
        mutation_scale = 0.5  # Initial mutation scale
        
        for _ in range(self.budget):
            candidate_solution = best_solution + mutation_scale * np.random.uniform(-0.5, 0.5, self.dim)  
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                mutation_scale *= 1.01  # Adaptive mutation scaling factor
            else:
                mutation_scale *= 0.99
        
        return best_solution