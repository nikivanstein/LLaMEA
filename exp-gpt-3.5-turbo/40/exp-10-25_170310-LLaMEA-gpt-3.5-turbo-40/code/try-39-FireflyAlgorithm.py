import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.alpha = 0.2  # alpha parameter for light absorption coefficient
        self.beta0 = 1.0  # initial attractiveness base value
        self.gamma = 1.0  # gamma parameter for attraction coefficient
        self.scale_factor = 0.5  # scaling factor for Lévy Flight step size

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness = self.beta0 * np.exp(-self.gamma * np.linalg.norm(self.population[j] - self.population[i])**2)
                        step = self.scale_factor * np.random.standard_cauchy(self.dim)
                        self.population[i] += attractiveness * (self.population[j] - self.population[i]) + self.alpha * step
        final_fitness = [func(x) for x in self.population]
        best_idx = np.argmin(final_fitness)
        best_solution = self.population[best_idx]

        return best_solution