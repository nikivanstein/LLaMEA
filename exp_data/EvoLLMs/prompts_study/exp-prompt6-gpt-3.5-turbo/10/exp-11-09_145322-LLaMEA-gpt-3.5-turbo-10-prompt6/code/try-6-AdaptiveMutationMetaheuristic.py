import numpy as np

class AdaptiveMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.ones(dim)
        
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution + self.mutation_rates * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            
            # Update mutation rates for each dimension independently
            self.mutation_rates = np.minimum(1.0, self.mutation_rates + 0.1 * (best_solution - candidate_solution) / (5.0 * self.budget))
        
        return best_solution