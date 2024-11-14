import numpy as np

class DynamicMutationEnhancedAdaptiveRandomSearch(EnhancedAdaptiveRandomSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        initial_solutions = self.initialize_population(5, self.dim)
        best_solution = min(initial_solutions, key=lambda x: func(x))
        best_fitness = func(best_solution)
        step_size = 1.0
        mutation_rate = self.mutation_rate

        for _ in range(self.budget):
            candidate_solution = best_solution + step_size * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_solution = self.local_search(candidate_solution, func)
            candidate_fitness = func(candidate_solution)

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                step_size *= self.alpha
                mutation_rate = max(0.01, mutation_rate * 0.9)  # Dynamic mutation rate adjustment
            else:
                step_size *= self.beta
                mutation_rate = min(0.5, mutation_rate * 1.1)

        return best_solution