import numpy as np

class ImprovedMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_step = 0.5  # Initial mutation step size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        step_size_history = [0.5, 0.4, 0.3]  # Multi-step mutation sizes
        
        for _ in range(self.budget):
            candidate_solution = best_solution + self.mutation_step * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
            
            # Adaptive mutation step size adjustment with multi-step sizes
            if np.random.rand() < 0.1:  # Probability of step size adjustment
                step_idx = min(len(step_size_history) - 1, int(np.random.uniform(0, len(step_size_history))))
                self.mutation_step *= step_size_history[step_idx]

        return best_solution