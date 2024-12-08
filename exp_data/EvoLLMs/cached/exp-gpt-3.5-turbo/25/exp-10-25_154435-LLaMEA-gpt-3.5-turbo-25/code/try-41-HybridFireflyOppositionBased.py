import numpy as np

class HybridFireflyOppositionBased:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            for _ in range(self.budget):
                solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                fitness = func(solution)

                if fitness < best_fitness:
                    best_solution = solution
                    best_fitness = fitness
                else:
                    # Opposition-based learning
                    opposite_solution = self.lower_bound + self.upper_bound - solution
                    opposite_fitness = func(opposite_solution)

                    if opposite_fitness < best_fitness:
                        best_solution = opposite_solution
                        best_fitness = opposite_fitness

        return best_solution