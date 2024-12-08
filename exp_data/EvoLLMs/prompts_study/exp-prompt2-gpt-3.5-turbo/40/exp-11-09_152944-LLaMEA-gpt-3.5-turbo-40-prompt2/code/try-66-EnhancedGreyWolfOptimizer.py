import numpy as np

class EnhancedGreyWolfOptimizer(GreyWolfOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_budget = budget

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

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
                    population[i] = (X1 + X2 + X3) / 3

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]
                
                # Dynamic population resizing
                if self.budget > self.initial_budget / 2 and _ % (self.budget // 10) == 0:
                    population = np.vstack([population, np.random.uniform(self.lb, self.ub, (self.budget // 5, self.dim))])
                    self.budget += self.budget // 5

            return alpha

        return optimize()