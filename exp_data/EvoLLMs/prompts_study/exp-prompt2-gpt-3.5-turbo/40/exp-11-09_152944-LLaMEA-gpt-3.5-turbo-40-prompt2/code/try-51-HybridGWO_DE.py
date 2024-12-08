import numpy as np

class HybridGWO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.5  # Differential Evolution scaling factor

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def de_mutate(population, current_index):
            candidates = [idx for idx in range(self.budget) if idx != current_index]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            return np.clip(a + self.F * (b - c), self.lb, self.ub)

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    mutant = de_mutate(population, i)
                    population[i] = (X1 + X2 + X3 + mutant) / 4

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()