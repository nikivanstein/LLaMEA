import numpy as np

class ImprovedDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_iterations = budget // self.population_size
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.9
        self.w_min = 0.4
        self.w_max = 0.9
        self.cr = 0.9
        self.f = 0.9

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim)), np.zeros(self.population_size), np.zeros(self.population_size)

        def mutate(x, a, b, c):
            return np.clip(x + self.c1 * (a - x) + self.c2 * (b - c), -5.0, 5.0)

        def improved_differential_evolution(population, fitness, best):
            new_population = np.copy(population)
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
                x_new = mutate(population[i], a, b, c)
                fitness_new = func(x_new)
                if fitness_new < fitness[i]:
                    new_population[i] = x_new
                    fitness[i] = fitness_new
                    if fitness_new < best:
                        best = fitness_new
            return new_population, fitness, best

        population, fitness, best = initialize_population()
        for _ in range(self.max_iterations):
            population, fitness, best = improved_differential_evolution(population, fitness, best)
        return best