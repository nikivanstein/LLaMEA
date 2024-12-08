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

        def differential_evolution(population, fitness, best, f, cr, success_rates):
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
                        success_rates[i] += 1  # Update success count
                if np.random.rand() < self.mutation_prob:
                    new_population[i] = explore_mutate(new_population[i])
            return new_population, fitness, best, success_rates

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)
        f = 0.9
        cr = 0.9
        success_rates = np.zeros(self.population_size)

        for _ in range(self.max_iterations):
            population, fitness, best, success_rates = differential_evolution(population, fitness, best, f, cr, success_rates)
            f = max(0.1, f * (1 + 0.1 * (np.sum(success_rates) / len(success_rates) - 0.2)))  # Adaptive mutation rate adjustment based on success
            cr = max(0.1, cr * (1 + 0.1 * (np.sum(success_rates) / len(success_rates) - 0.2)))  # Adaptive crossover rate adjustment based on success
        return best