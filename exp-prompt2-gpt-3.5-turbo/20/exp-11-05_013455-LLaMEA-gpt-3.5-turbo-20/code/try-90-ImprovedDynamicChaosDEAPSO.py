import numpy as np

class ImprovedDynamicChaosDEAPSO:
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

        def chaotic_search(x, best, chaos_param):
            new_x = x + chaos_param * np.random.uniform(-5.0, 5.0, x.shape)
            new_x = np.clip(new_x, -5.0, 5.0)
            if func(new_x) < func(x):
                return new_x
            else:
                return x

        def local_search(x, best, radius=0.1):
            x_new = np.clip(x + radius * np.random.normal(0, 1, x.shape), -5.0, 5.0)
            if func(x_new) < func(x):
                return x_new
            else:
                return x

        def differential_evolution(population, fitness, best, f, cr, chaos_param):
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
                    new_population[i] = self_adaptive_mutate(new_population[i], f * (1 - (fitness_new - best) / best))
                new_population[i] = chaotic_search(new_population[i], best, chaos_param)
                new_population[i] = local_search(new_population[i], best)  # Integrate local search
            return new_population, fitness, best

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)
        f = 0.9
        cr = 0.9
        chaos_param = 0.3

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best, f, cr, chaos_param)
            f = max(0.1, f * 0.95)  # Adaptive mutation rate adjustment
            cr = max(0.1, cr * 0.95)  # Adaptive crossover rate adjustment
            diversity = np.std(population, axis=0)
            chaos_param = max(0.1, min(0.5, np.mean(diversity)))
            chaos_param = max(0.1, chaos_param * 0.97)  # Dynamic chaos parameter adjustment based on diversity
        return best