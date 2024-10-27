import numpy as np

class EHEA:
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

        def de_mutate(x_r1, x_r2, x_r3, F=0.5):
            return x_r1 + F * (x_r2 - x_r3)

        def pso_update(x, p_best, g_best, w=0.5, c1=1.5, c2=1.5):
            v = w * x['velocity'] + c1 * np.random.rand() * (p_best - x['position']) + c2 * np.random.rand() * (g_best - x['position'])
            x['position'] = np.clip(x['position'] + v, -5.0, 5.0)
            x['velocity'] = v
            return x

        population = initialize_population()
        fitness_values = np.array([objective_function(ind) for ind in population])
        p_best_idx = np.argmin(fitness_values)
        g_best_idx = p_best_idx
        g_best = population[g_best_idx].copy()

        for _ in range(self.max_iter):
            for idx, ind in enumerate(population):
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                new_ind = de_mutate(r1, r2, r3)
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
                    population[i] = pso_update({'position': population[i], 'velocity': np.zeros(self.dim)}, p_best, g_best)

        return g_best