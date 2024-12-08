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

        def mutation(population, F=0.5):
            mutated_pop = np.zeros_like(population)
            for i in range(self.budget):
                r1, r2, r3 = np.random.choice(self.budget, 3, replace=False)
                mutated_pop[i] = population[r1] + F * (population[r2] - population[r3])
            return mutated_pop

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                mutated_population = mutation(population)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3

                pop_combined = np.vstack((population, mutated_population))
                population = pop_combined[np.argsort([func(ind) for ind in pop_combined])[:self.budget]]
                alpha, beta, delta = population[:3]

            return alpha

        return optimize()