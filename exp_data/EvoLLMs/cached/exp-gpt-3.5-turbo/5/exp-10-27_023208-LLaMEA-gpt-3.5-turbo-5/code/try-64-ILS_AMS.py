import numpy as np

class ILS_AMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_strength = 0.1
        self.adaptive_factor = 0.9

    def _mutation(self, current_solution):
        return current_solution + self.mutation_strength * np.random.uniform(-1, 1, self.dim)

    def _optimize_func(self, func, current_solution):
        best_solution = current_solution
        for _ in range(self.budget):
            candidate_solution = self._mutation(current_solution)
            if func(candidate_solution) < func(current_solution):
                current_solution = candidate_solution
                if func(candidate_solution) < func(best_solution):
                    best_solution = candidate_solution
            self.mutation_strength *= self.adaptive_factor
        return best_solution

    def __call__(self, func):
        initial_solution = np.random.uniform(-5, 5, self.dim)
        return self._optimize_func(func, initial_solution)