import numpy as np

class AdaptiveStepSizeMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.steps = 5

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            step_size = 1.0  # Initial step size
            for _ in range(self.steps):
                candidate_solution = best_solution + self.mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors * step_size
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
                candidate_fitness = func(candidate_solution)

                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness

                # Update step size based on individual and population performance
                step_size *= 0.9 if candidate_fitness < best_fitness else 1.1
                step_size = np.clip(step_size, 0.1, 2.0)

            # Update mutation rate based on population diversity
            self.mutation_rate *= 0.99

        return best_solution