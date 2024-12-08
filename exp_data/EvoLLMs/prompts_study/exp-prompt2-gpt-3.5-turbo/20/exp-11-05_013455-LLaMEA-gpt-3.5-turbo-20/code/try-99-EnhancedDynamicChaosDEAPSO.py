import numpy as np

class EnhancedDynamicChaosDEAPSO(DynamicChaosDEAPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_control = 0.1

    def __call__(self, func):
        def self_adaptive_mutate(x, f):
            return np.clip(x + f * np.random.normal(0, 1, x.shape), -5.0, 5.0)

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
                    new_population[i] = self_adaptive_mutate(new_population[i], f)
                new_population[i] = chaotic_search(new_population[i], best, chaos_param)
                new_population[i] = local_search(new_population[i], best)
            return new_population, fitness, best

        self.mutation_prob = self.mutation_control  # Adjusted mutation probability based on individual performance
        return super().__call__(func)