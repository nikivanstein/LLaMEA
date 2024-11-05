import numpy as np

class DynamicInertiaEnhancedDEAPSO(EnhancedDEAPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.w_dynamic = self.w

    def __call__(self, func):
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
            self.mutation_prob = max(0.1, self.mutation_prob * 0.95)

            global_best_idx = np.argmin(fitness)
            for i in range(self.population_size):
                if i != global_best_idx:
                    self.w_dynamic = max(self.w_min, min(self.w_max, self.w_dynamic + np.random.normal(0, 0.1)))
                    new_population[i] = self.w_dynamic * new_population[i] + (1 - self.w_dynamic) * population[i]

            return new_population, fitness, best

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best)
        return best