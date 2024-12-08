import numpy as np

class CrowdingEnhancedMetaheuristicAlgorithm:
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

            candidate_solution = best_solution + np.random.uniform(-self.mutation_scale, self.mutation_scale, self.dim)  # Mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            else:
                crowding_solution = best_solution + 0.3 * np.random.rand(self.dim)  # Introduce crowding-based solution
                crowding_solution = np.clip(crowding_solution, -5.0, 5.0)  # Ensure crowding solution is within bounds
                crowding_fitness = func(crowding_solution)
                
                if crowding_fitness < best_fitness:
                    best_solution = crowding_solution
                    best_fitness = crowding_fitness
        
        return best_solution