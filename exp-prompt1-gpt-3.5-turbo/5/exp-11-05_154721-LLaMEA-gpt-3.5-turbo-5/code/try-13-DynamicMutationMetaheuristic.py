import numpy as np

class DynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_step_size = 0.5  # Initial mutation step size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        mutation_step_size = self.initial_step_size
        
        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-mutation_step_size, mutation_step_size, self.dim)  # Dynamic mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                mutation_step_size *= 0.99  # Decrease mutation step size for exploitation
            else:
                mutation_step_size *= 1.01  # Increase mutation step size for exploration
        
        return best_solution