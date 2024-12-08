import numpy as np

class ParallelSearchMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget//5):  # Modified for parallel search
            mutation_strength = np.random.uniform(0, 1)
            candidate_solutions = [best_solution + mutation_strength * np.random.standard_normal(self.dim) for _ in range(5)]  # Generate 5 candidate solutions
            candidate_fitnesses = [func(candidate) for candidate in candidate_solutions]
            
            best_candidate_idx = np.argmin(candidate_fitnesses)
            if candidate_fitnesses[best_candidate_idx] < best_fitness:
                best_solution = candidate_solutions[best_candidate_idx]
                best_fitness = candidate_fitnesses[best_candidate_idx]
            
            self.mutation_rate = 0.9 * self.mutation_rate + 0.1 * abs(best_fitness - candidate_fitnesses[best_candidate_idx])
        
        return best_solution