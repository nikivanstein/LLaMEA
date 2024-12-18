class ImprovedDynamicPopulationSizeHybridFireAntCuckooOptimizationImproved(DynamicPopulationSizeHybridFireAntCuckooOptimizationImproved):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.chaos_param = 0.3

    def _chaotic_map(self, x):
        return np.sin(x) * np.cos(x) * self.chaos_param

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        population_size = 10

        for _ in range(self.budget):
            if np.random.rand() < mutation_prob:
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
                mutation_prob *= 0.99
                self.step_size *= 0.99

            steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])

            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.98

            global_best_solution = new_solutions[np.argmin(new_fitnesses)]
            chaotic_map = self._chaotic_map(best_solution)
            combined_solution = best_solution + self.step_size * (global_best_solution - best_solution) * chaotic_map
            combined_solution = np.clip(combined_solution, self.lower_bound, self.upper_bound)
            combined_fitness = func(combined_solution)

            if combined_fitness < best_fitness:
                best_solution = combined_solution
                best_fitness = combined_fitness

        return best_solution