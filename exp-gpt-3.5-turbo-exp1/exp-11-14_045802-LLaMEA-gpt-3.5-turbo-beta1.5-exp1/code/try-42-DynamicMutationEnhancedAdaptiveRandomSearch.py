import numpy as np

class DynamicMutationEnhancedAdaptiveRandomSearch(EnhancedAdaptiveRandomSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.dynamic_mutation_rate = 0.1

    def dynamic_local_search(self, center, func, mutation_rate):
        neighborhood = 0.1
        new_center = center.copy()
        for i in range(self.dim):
            new_center[i] += np.random.uniform(-neighborhood, neighborhood)
            new_center[i] = np.clip(new_center[i], -5.0, 5.0)
        
        if func(new_center) < func(center):
            return new_center, mutation_rate * self.alpha
        else:
            return center, mutation_rate * self.beta

    def __call__(self, func):
        initial_solutions = self.initialize_population(5, self.dim)
        best_solution = min(initial_solutions, key=lambda x: func(x))
        best_fitness = func(best_solution)
        step_size = 1.0
        mutation_rate = self.dynamic_mutation_rate

        for _ in range(self.budget):
            candidate_solution = best_solution + step_size * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_solution, mutation_rate = self.dynamic_local_search(candidate_solution, func, mutation_rate)
            candidate_fitness = func(candidate_solution)

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                step_size *= self.alpha
            else:
                step_size *= self.beta

        return best_solution