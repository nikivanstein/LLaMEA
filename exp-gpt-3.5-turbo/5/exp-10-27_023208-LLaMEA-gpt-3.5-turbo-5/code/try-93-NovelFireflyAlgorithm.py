import numpy as np

class NovelFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0

    def _attractiveness(self, x, y):
        return np.exp(-np.linalg.norm(x - y))

    def _update_position(self, x, y, beta):
        return x + beta * (y - x) + self.alpha * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        for _ in range(self.budget):
            for i, firefly in enumerate(population):
                for j, nearby_firefly in enumerate(population):
                    if func(nearby_firefly) < func(firefly):
                        attractiveness = self._attractiveness(firefly, nearby_firefly)
                        beta = self.beta0 * np.exp(-0.2 * _ / self.budget)
                        new_position = self._update_position(firefly, nearby_firefly, beta)
                        if func(new_position) < func(firefly):
                            population[i] = new_position
            best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution

novel_firefly_algorithm = NovelFireflyAlgorithm(budget=1000, dim=10)