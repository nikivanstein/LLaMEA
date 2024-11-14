class EnhancedDynamicAdaptiveRandomSearch(ImprovedEnhancedAdaptiveRandomSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.gamma = 0.9

    def __call__(self, func):
        initial_solutions = self.initialize_population(5, self.dim)
        best_solution = min(initial_solutions, key=lambda x: func(x))
        best_fitness = func(best_solution)
        step_size = 1.0

        for _ in range(self.budget):
            candidate_solution = best_solution + step_size * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_solution = self.local_search(candidate_solution, func)
            candidate_fitness = func(candidate_solution)

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                improvement_rate = (best_fitness - candidate_fitness) / best_fitness
                step_size *= (1 + self.gamma * improvement_rate)
            else:
                step_size *= self.beta

        return best_solution