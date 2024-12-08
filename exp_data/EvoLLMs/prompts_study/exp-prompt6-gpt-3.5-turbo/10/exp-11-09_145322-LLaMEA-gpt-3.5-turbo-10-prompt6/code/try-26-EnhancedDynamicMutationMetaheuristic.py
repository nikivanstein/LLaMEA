import numpy as np

class EnhancedDynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.global_best_solution = np.random.uniform(-5.0, 5.0, dim)
    
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            candidate_solution = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_fitness = func(candidate_solution)

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness

            # Update mutation factors based on individual performance
            self.mutation_factors *= 0.9 if candidate_fitness < best_fitness else 1.1
            self.mutation_factors = np.clip(self.mutation_factors, 0.1, 2.0)

            # Update mutation rate based on a combination of individual performance and global search strategy
            self.mutation_rate *= 0.99 if np.all(np.abs(best_solution - self.global_best_solution) > 1e-6) else 1.01

            # Update global best solution
            if candidate_fitness < func(self.global_best_solution):
                self.global_best_solution = np.copy(candidate_solution)

        return best_solution