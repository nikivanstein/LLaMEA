import numpy as np

class NovelDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5
        self.CR = 0.9
        self.pop_size = 10
        self.min_pop_size = 5
        self.max_pop_size = 20

    def __call__(self, func):
        def bound_check(x):
            return np.clip(x, -5.0, 5.0)

        def mutation(x_r1, x_r2, x_r3):
            return bound_check(x_r1 + self.F * (x_r2 - x_r3))

        def crossover(x, v):
            j_rand = np.random.randint(self.dim)
            u = np.array([v[i] if i == j_rand or np.random.rand() < self.CR else x[i] for i in range(self.dim)])
            return bound_check(u)

        def evolve_population(population):
            new_population = []
            for i, x in enumerate(population):
                idxs = [idx for idx in range(len(population)) if idx != i]
                x_r1, x_r2, x_r3 = population[np.random.choice(idxs, 3, replace=False)]
                v = mutation(x_r1, x_r2, x_r3)
                u = crossover(x, v)
                new_population.append(u)
            return new_population

        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.pop_size)]
        fitness_values = [func(x) for x in population]

        for _ in range(self.budget // self.pop_size):
            population = evolve_population(population)
            new_fitness_values = [func(x) for x in population]
            for i in range(self.pop_size):
                if new_fitness_values[i] < fitness_values[i]:
                    fitness_values[i] = new_fitness_values[i]

        return population[np.argmin(fitness_values)]