import numpy as np

class AdaptiveMutationDynamicMultiStepMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.steps = 5
        self.performance_history = []

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            for _ in range(self.steps):
                candidate_solution = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
                candidate_fitness = func(candidate_solution)

                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness

                self.performance_history.append(candidate_fitness)

                # Update mutation factors based on historical performance
                if len(self.performance_history) > self.dim:
                    last_dim_fitness = self.performance_history[-self.dim:]
                    self.mutation_factors = np.where(last_dim_fitness[-1] < last_dim_fitness, self.mutation_factors * 0.9, self.mutation_factors * 1.1)
                    self.mutation_factors = np.clip(self.mutation_factors, 0.1, 2.0)

            # Update mutation rate based on population diversity
            self.mutation_rate *= 0.99

        return best_solution