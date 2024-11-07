import numpy as np

class EnhancedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-0.5, 0.5, self.dim)  # Mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            else:
                # Local search step for exploitation
                for _ in range(3):
                    local_candidate = best_solution + np.random.uniform(-0.1, 0.1, self.dim)
                    local_candidate = np.clip(local_candidate, -5.0, 5.0)
                    local_fitness = func(local_candidate)
                    if local_fitness < best_fitness:
                        best_solution = local_candidate
                        best_fitness = local_fitness
                        break
        
        return best_solution