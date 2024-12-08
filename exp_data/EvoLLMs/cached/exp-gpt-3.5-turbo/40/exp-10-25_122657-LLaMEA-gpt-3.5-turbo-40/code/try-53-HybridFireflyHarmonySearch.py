import numpy as np

class HybridFireflyHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def evaluate_population(population):
            return np.array([func(solution) for solution in population])

        def update_population(population, fitness):
            new_population = np.copy(population)
            for i in range(self.budget):
                for j in range(self.dim):
                    if np.random.rand() < 0.4:
                        new_population[i][j] = (population[i][j] + np.mean(population[:, j])) / 2
            return new_population

        population = initialize_population()
        fitness = evaluate_population(population)

        for _ in range(self.budget - self.budget // 10):
            population = update_population(population, fitness)
            fitness = evaluate_population(population)

        best_idx = np.argmin(fitness)
        return population[best_idx]