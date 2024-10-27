import numpy as np

class AdaptiveDEWithLocalSearch:
    def __init__(self, budget, dim, pop_size=50, mutation_factor=0.8, crossover_prob=0.9, local_search_iter=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.local_search_iter = local_search_iter

    def __call__(self, func):
        def local_search(x):
            best_x = x.copy()
            best_fitness = func(x)
            for _ in range(self.local_search_iter):
                new_x = x + np.random.normal(0, 0.1, size=self.dim)
                new_fitness = func(new_x)
                if new_fitness < best_fitness:
                    best_x = new_x
                    best_fitness = new_fitness
            return best_x

        population = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        population_fitness = np.array([func(p) for p in population])
        best_idx = np.argmin(population_fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                target = population[i]
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)

                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, target)

                trial = local_search(trial)
                trial_fitness = func(trial)

                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

                    if trial_fitness < func(best_solution):
                        best_solution = trial

        return best_solution