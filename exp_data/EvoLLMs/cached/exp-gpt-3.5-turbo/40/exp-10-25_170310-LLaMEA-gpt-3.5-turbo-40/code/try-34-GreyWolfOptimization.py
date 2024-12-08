import numpy as np

class GreyWolfOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.a = 2  # Alpha
        self.a_damp = 2 * np.log(2)
        self.max_iter = 1000

    def __call__(self, func):
        for t in range(self.max_iter):
            for i in range(self.budget):
                alpha_pos = self.population[np.argmin([func(x) for x in self.population])]
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * self.a * r1 - self.a
                    C1 = 2 * r2
                    D_alpha = np.abs(C1 * alpha_pos[j] - self.population[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    self.population[i][j] = np.clip(X1, -5.0, 5.0)
            self.a = self.a - (self.a / self.a_damp)  # Update alpha over iterations

        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution