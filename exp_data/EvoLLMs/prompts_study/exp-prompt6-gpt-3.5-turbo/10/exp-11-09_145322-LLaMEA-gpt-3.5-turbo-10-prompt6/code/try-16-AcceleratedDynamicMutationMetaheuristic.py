class AcceleratedDynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.prev_best_fitness = None
        self.prev_best_solution = None

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            candidate_solution = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_fitness = func(candidate_solution)

            if self.prev_best_fitness is not None:
                self.mutation_factors *= 0.9 if candidate_fitness < self.prev_best_fitness else 1.1
                self.mutation_factors = np.clip(self.mutation_factors, 0.1, 2.0)

            if self.prev_best_solution is not None:
                self.mutation_rate *= 0.99 if np.linalg.norm(candidate_solution - self.prev_best_solution) < 0.1 else 1.01

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness

            self.prev_best_fitness = best_fitness
            self.prev_best_solution = best_solution

        return best_solution