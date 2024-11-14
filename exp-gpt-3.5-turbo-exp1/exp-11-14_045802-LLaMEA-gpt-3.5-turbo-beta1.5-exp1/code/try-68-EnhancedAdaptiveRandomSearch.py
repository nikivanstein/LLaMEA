class EnhancedAdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 1.2
        self.beta = 0.8
        self.mutation_rate = 0.1
        self.min_mutation_rate = 0.01
        self.max_mutation_rate = 0.5

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
                step_size *= self.alpha
                self.mutation_rate = min(self.mutation_rate * 1.1, self.max_mutation_rate)
            else:
                step_size *= self.beta
                self.mutation_rate = max(self.mutation_rate * 0.9, self.min_mutation_rate)

        return best_solution