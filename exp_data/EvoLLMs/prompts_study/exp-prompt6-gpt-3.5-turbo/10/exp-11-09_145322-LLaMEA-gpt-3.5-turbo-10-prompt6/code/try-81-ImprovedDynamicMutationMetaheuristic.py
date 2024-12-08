import numpy as np

class ImprovedDynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0
        self.mutation_factors = np.ones(dim)
        self.min_step = 0.1
        self.max_step = 10.0

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            improvement_count = 0
            for i in range(self.dim):
                candidate_solution = best_solution.copy()
                candidate_solution[i] += self.mutation_rate * np.random.uniform(-1, 1) * self.mutation_factors[i]
                candidate_solution[i] = np.clip(candidate_solution[i], -5.0, 5.0)
                candidate_fitness = func(candidate_solution)

                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
                    improvement_count += 1

                # Adjust mutation factor based on individual performance
                self.mutation_factors[i] *= 0.9 if candidate_fitness < best_fitness else 1.1
                self.mutation_factors[i] = np.clip(self.mutation_factors[i], 0.1, 2.0)

            # Update mutation rate based on improvement count
            self.mutation_rate *= 0.99 if improvement_count > 0 else 1.01
            self.mutation_rate = np.clip(self.mutation_rate, self.min_step, self.max_step)

        return best_solution