import numpy as np

class ElitistEvolutionStrategy:
    def __init__(self, budget, dim, mu=5, lambda_=20, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf

        for _ in range(self.budget // self.lambda_):
            offspring = [np.random.randn(self.dim) * self.sigma for _ in range(self.lambda_)]
            solutions = [best_solution + offspring[i] if np.random.rand() < 0.2 else np.random.rand(self.dim) * 10 - 5 for i in range(self.lambda_)]
            fitness_values = [func(sol) for sol in solutions]

            for i in range(self.lambda_):
                if fitness_values[i] < best_fitness:
                    best_fitness = fitness_values[i]
                    best_solution = solutions[i]

        return best_solution