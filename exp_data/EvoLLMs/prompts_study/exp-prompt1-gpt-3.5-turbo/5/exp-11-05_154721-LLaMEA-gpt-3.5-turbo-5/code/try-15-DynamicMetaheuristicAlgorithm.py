import numpy as np

class DynamicMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.step_size = 0.5  # Initial step size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-self.step_size, self.step_size, self.dim)  # Mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                self.step_size *= 0.99  # Dynamic step size adaptation
        
        return best_solution