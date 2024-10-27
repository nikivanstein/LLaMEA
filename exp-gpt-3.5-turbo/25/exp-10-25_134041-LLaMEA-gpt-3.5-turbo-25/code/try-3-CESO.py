import numpy as np

class CESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def crossover(x1, x2):
            alpha = np.random.uniform(0, 1, size=self.dim)
            child = x1 * alpha + x2 * (1 - alpha)
            return child

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        p_best_idx = np.argmin(fitness_values)
        g_best_idx = p_best_idx
        g_best = population[g_best_idx].copy()

        for _ in range(self.max_iter):
            for idx, ind in enumerate(population):
                r1, r2 = np.random.choice(population, 2, replace=False)
                new_ind = crossover(ind, r1)
                new_ind_fitness = objective_function(new_ind)

                if new_ind_fitness < fitness_values[idx]:
                    population[idx] = new_ind
                    fitness_values[idx] = new_ind_fitness

                if new_ind_fitness < fitness_values[p_best_idx]:
                    p_best_idx = idx

                p_best = population[p_best_idx]
                g_best_fitness = objective_function(g_best)

                if new_ind_fitness < g_best_fitness:
                    g_best = new_ind

            for i in range(self.population_size):
                r1, r2 = np.random.choice(population, 2, replace=False)
                population[i] = crossover(population[i], r1)

        return g_best