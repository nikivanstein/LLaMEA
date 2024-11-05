import numpy as np

class EnhancedDynamicChaosDEAPSO(DynamicChaosDEAPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.radius_min = 0.05
        self.radius_max = 0.2

    def local_search(self, x, best, radius=0.1):
        radius = np.clip(radius, self.radius_min, self.radius_max)
        x_new = np.clip(x + radius * np.random.normal(0, 1, x.shape), -5.0, 5.0)
        if func(x_new) < func(x):
            return x_new
        else:
            return x

    def adaptive_local_search(self, x, best, radius):
        return self.local_search(x, best, radius)

    def differential_evolution(self, population, fitness, best, f, cr, chaos_param):
        new_population = np.copy(population)
        for i in range(self.population_size):
            a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
            x_new = self.mutate(population[i], a, b, c, f)
            if np.all(x_new == population[i]) or np.random.rand() < cr:
                x_new = a + f * (b - c)
            fitness_new = func(x_new)
            if fitness_new < fitness[i]:
                new_population[i] = x_new
                fitness[i] = fitness_new
                if fitness_new < best:
                    best = fitness_new
            if np.random.rand() < self.mutation_prob:
                new_population[i] = self.self_adaptive_mutate(new_population[i], f)
            new_population[i] = self.chaotic_search(new_population[i], best, chaos_param)
            new_population[i] = self.adaptive_local_search(new_population[i], best, self.radius_min + (self.radius_max - self.radius_min) * (i+1) / self.population_size)  # Integrate adaptive local search
        return new_population, fitness, best