import numpy as np

class RefinedCuckooSearchAlgorithm:
    def __init__(self, budget, dim, population_size=10, alpha=0.9, pa=0.25):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.pa = pa
        self.population = np.random.uniform(-5.0, 5.0, (population_size, dim))
        self.best_solution = np.copy(self.population[0])

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step + 0.1 * np.random.normal(0, 1, size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            nest_idx = np.random.randint(self.population_size)
            new_nest = self.population[nest_idx] + self.levy_flight()
            if func(new_nest) < func(self.population[nest_idx]):
                self.population[nest_idx] = new_nest
            replace_idx = np.random.randint(self.population_size)
            if func(new_nest) < func(self.population[replace_idx]):
                self.population[replace_idx] = new_nest
            self.population = self.population[np.argsort([func(x) for x in self.population])]
            self.population[-1] = np.random.uniform(-5.0, 5.0, self.dim)
            if func(self.population[0]) < func(self.best_solution):
                self.best_solution = np.copy(self.population[0])
        return self.best_solution