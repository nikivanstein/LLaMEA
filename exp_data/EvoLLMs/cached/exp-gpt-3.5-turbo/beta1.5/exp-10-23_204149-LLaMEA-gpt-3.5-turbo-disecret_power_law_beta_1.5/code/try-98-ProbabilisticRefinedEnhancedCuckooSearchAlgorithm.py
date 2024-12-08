import numpy as np

class ProbabilisticRefinedEnhancedCuckooSearchAlgorithm:
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
            new_nest_fitness = func(new_nest)
            nest_fitness = func(self.population[nest_idx])
            if new_nest_fitness < nest_fitness:
                self.population[nest_idx] = new_nest
            if np.random.rand() < 0.02631578947368421:  # Probabilistic individual line refinement
                self.population[nest_idx] += np.random.uniform(-0.1, 0.1, self.dim)
            replace_idx = np.random.randint(self.population_size)
            replace_fitness = func(self.population[replace_idx])
            if new_nest_fitness < replace_fitness:
                self.population[replace_idx] = new_nest
            self.population = self.population[np.argsort([func(x) for x in self.population])]
            self.population[-1] = np.random.uniform(-5.0, 5.0, self.dim)
            best_fitness = func(self.population[0])
            if best_fitness < func(self.best_solution):
                self.best_solution = np.copy(self.population[0])
        return self.best_solution