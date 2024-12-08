import numpy as np

class AdaptiveMutationMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        mutation_rate = 0.5

        for _ in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-mutation_rate, mutation_rate, self.dim)  # Mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                mutation_rate *= 0.95  # Decrease mutation rate for exploitation
            else:
                mutation_rate *= 1.05  # Increase mutation rate for exploration
        
        return best_solution