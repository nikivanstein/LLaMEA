import numpy as np

class AdaptiveMutationStepMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.base_steps = 5
        self.min_steps = 1
        self.max_steps = 10

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            adaptive_steps = int(self.base_steps * (1 + 0.1 * np.sin(_ / self.budget * np.pi)))  # Adaptive step size control
            for _ in range(adaptive_steps):
                candidate_solution = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
                candidate_fitness = func(candidate_solution)

                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness

                # Update mutation factors based on individual performance
                self.mutation_factors *= 0.9 if candidate_fitness < best_fitness else 1.1
                self.mutation_factors = np.clip(self.mutation_factors, 0.1, 2.0)

            # Update mutation rate based on population diversity
            self.mutation_rate *= 0.99

        return best_solution