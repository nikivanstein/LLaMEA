import numpy as np

class EnhancedDynamicChaosDEAPSO:
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

        def self_adaptive_mutate(x, f):
            return np.clip(x + f * np.random.normal(0, 1, x.shape), -5.0, 5.0)

        def adaptive_chaotic_search(x, best, diversity):
            chaos_param = 0.3 + np.mean(diversity) / 20.0  # Adjust chaos based on diversity
            new_x = x + chaos_param * np.random.uniform(-5.0, 5.0, x.shape)
            new_x = np.clip(new_x, -5.0, 5.0)
            if func(new_x) < func(x):
                return new_x
            else:
                return x

        def differential_evolution(population, fitness, best, f, cr, diversity):
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
                    new_population[i] = self_adaptive_mutate(new_population[i], f)
                new_population[i] = adaptive_chaotic_search(new_population[i], best, diversity)
            return new_population, fitness, best

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)
        f = 0.9
        cr = 0.9

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best, f, cr, np.std(population, axis=0))
            f = max(0.1, f * 0.95)  # Adaptive mutation rate adjustment
            cr = max(0.1, cr * 0.95)  # Adaptive crossover rate adjustment
        return best