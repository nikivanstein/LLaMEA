import numpy as np

class EnhancedDynamicChaosDEAPSO(DynamicChaosDEAPSO):
    def chaotic_search(x, best, chaos_param):
            new_x = x + chaos_param * np.random.uniform(-5.0, 5.0, x.shape)
            new_x = np.clip(new_x, -5.0, 5.0)
            if func(new_x) < func(x):
                return new_x
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
                    new_population[i] = self_adaptive_mutate(new_population[i], f)
                new_population[i] = chaotic_search(new_population[i], best, chaos_param)
            return new_population, fitness, best