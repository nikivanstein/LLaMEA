import numpy as np

class EnhancedGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]
            step_size = np.full(self.budget, 0.5)  # Initialize step size for each individual

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * step_size[i] * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * step_size[i] * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * step_size[i] * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3

                    # Update step size based on individual performance
                    if func(population[i]) < func(x):
                        step_size[i] *= 1.1  # Increase step size for better individuals
                    else:
                        step_size[i] *= 0.9  # Decrease step size for worse individuals

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()