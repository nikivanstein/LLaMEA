import numpy as np

class ImprovedDynamicDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_iterations = budget // self.population_size
        self.w = 0.9
        self.w_min = 0.4
        self.w_max = 0.9
        self.mutation_prob = 0.2
        self.crossover_prob = 0.9

    def __call__(self, func):
        def mutate(x, a, b, c, f):
            return np.clip(a + f * (b - c), -5.0, 5.0)

        def explore_mutate(x):
            return np.clip(x + np.random.normal(0, 1, x.shape), -5.0, 5.0)

        def chaotic_search(x, best, dim):
            chaotic_param = 0.8
            for i in range(dim):
                x[i] = best + chaotic_param * np.random.standard_cauchy()
            return x

        def differential_evolution(population, fitness, best, f, cr):
            new_population = np.copy(population)
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
                x_new = mutate(population[i], a, b, c, f)
                if np.all(x_new == population[i]) or np.random.rand() < cr:
                    x_new = a + f * (b - c)
                fitness_new = func(x_new)
                if fitness_new < fitness[i]:
                    new_population[i] = x_new
                    fitness[i] = fitness_new
                    if fitness_new < best:
                        best = fitness_new
                if np.random.rand() < self.mutation_prob:
                    if np.random.rand() < 0.2:  # 20% chance of chaotic search
                        new_population[i] = chaotic_search(new_population[i], best, self.dim)
                    else:
                        new_population[i] = explore_mutate(new_population[i])
            return new_population, fitness, best

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)
        f = 0.9
        cr = 0.9

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best, f, cr)
            f = max(0.1, f * 0.95)  # Adaptive mutation rate adjustment
            cr = max(0.1, cr * 0.95)  # Adaptive crossover rate adjustment
        return best