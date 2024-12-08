import numpy as np

class EnhancedDEAPSO:
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
        self.mutation_prob = 0.2  # New parameter for mutation probability with adaptive adjustment

    def __call__(self, func):
        def mutate(x, a, b, c):
            return np.clip(a + self.f * (b - c), -5.0, 5.0)

        def explore_mutate(x):
            return np.clip(x + np.random.normal(0, 1, x.shape), -5.0, 5.0)

        def differential_evolution(population, fitness, best):
            new_population = np.copy(population)
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
                x_new = mutate(population[i], a, b, c)
                if np.all(x_new == population[i]) or np.random.rand() < self.cr:
                    x_new = a + self.f * (b - c)
                fitness_new = func(x_new)
                if fitness_new < fitness[i]:
                    new_population[i] = x_new
                    fitness[i] = fitness_new
                    if fitness_new < best:
                        best = fitness_new
                if np.random.rand() < self.mutation_prob:
                    new_population[i] = explore_mutate(new_population[i])
            self.mutation_prob = max(0.1, self.mutation_prob * 0.95)  # Adaptive mutation probability adjustment
            return new_population, fitness, best

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best)
        return best