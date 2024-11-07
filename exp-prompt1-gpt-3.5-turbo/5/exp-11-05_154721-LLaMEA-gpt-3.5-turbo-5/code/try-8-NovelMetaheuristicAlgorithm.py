import numpy as np

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        mutation_rate = 0.5
        
        for _ in range(self.budget):
            candidate_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim)  # Mutation step with dynamic rate
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                mutation_rate *= 0.9  # Adjust mutation rate based on fitness improvement
        
        return best_solution