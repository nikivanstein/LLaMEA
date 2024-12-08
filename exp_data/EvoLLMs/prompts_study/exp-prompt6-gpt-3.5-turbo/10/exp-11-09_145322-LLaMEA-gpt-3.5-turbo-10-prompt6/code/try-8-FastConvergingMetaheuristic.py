import numpy as np

class FastConvergingMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)

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
                self.mutation_factors = np.ones(self.dim)

            # Update mutation rate based on individual performance
            self.mutation_rate *= 0.99 if candidate_fitness < best_fitness else 1.01

        return best_solution