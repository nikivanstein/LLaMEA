import numpy as np

class AdaptiveMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.steps = 5

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        mutation_rate = self.base_mutation_rate

        for _ in range(self.budget):
            for _ in range(self.steps):
                candidate_solution = best_solution + mutation_rate * np.random.uniform(-1, 1, self.dim) * self.mutation_factors
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
                candidate_fitness = func(candidate_solution)

                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
                    mutation_rate *= 1.1  # Increase mutation rate upon fitness improvement
                else:
                    mutation_rate *= 0.9  # Decrease mutation rate if no improvement

                mutation_rate = np.clip(mutation_rate, 0.1, 2.0)  # Clip mutation rate within bounds

            mutation_rate *= 0.99  # Update mutation rate at the end of each iteration

        return best_solution