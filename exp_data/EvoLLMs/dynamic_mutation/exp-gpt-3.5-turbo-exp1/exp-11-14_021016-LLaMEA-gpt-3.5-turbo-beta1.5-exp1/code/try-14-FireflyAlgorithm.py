import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = None

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        def attractiveness(i, j, iter_num, max_iter):
            r = np.linalg.norm(self.population[i] - self.population[j])
            return 1 / (1 + r) * np.exp(-iter_num / max_iter)

        def move_fireflies(iter_num, max_iter):
            new_population = np.copy(self.population)
            for i in range(self.budget):
                alpha = 1.0 - iter_num / max_iter
                beta = 1.0 - iter_num / max_iter
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        new_population[i] += alpha * np.exp(-beta * np.linalg.norm(self.population[j] - self.population[i]) ** 2) * (self.population[j] - self.population[i]) + np.random.uniform(-1, 1, self.dim)
            self.population = new_population

        self.population = initialize_population()
        max_iterations = self.budget
        for iter_num in range(max_iterations):
            move_fireflies(iter_num, max_iterations)
        best_solution = min(self.population, key=lambda x: func(x))
        return best_solution