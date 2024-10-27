import numpy as np

class DifferentialEvolutionLocalSearch:
    def __init__(self, budget, dim, population_size=50, differential_weight=0.5, crossover_prob=0.9, local_search_iter=5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.differential_weight = differential_weight
        self.crossover_prob = crossover_prob
        self.local_search_iter = local_search_iter

    def __call__(self, func):
        def local_search(x):
            best_x = x.copy()
            best_fitness = func(x)
            for _ in range(self.local_search_iter):
                new_x = x + np.random.uniform(-0.1, 0.1, size=self.dim)
                new_fitness = func(new_x)
                if new_fitness < best_fitness:
                    best_x = new_x
                    best_fitness = new_fitness
            return best_x

        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        population_fitness = np.array([func(p) for p in population])
        best_idx = np.argmin(population_fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = population[i] + self.differential_weight * (a - b)
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, population[i])

                trial = np.clip(trial, -5.0, 5.0)
                
                trial = local_search(trial)
                trial_fitness = func(trial)

                if trial_fitness < population_fitness[i]:
                    population[i] = trial
                    population_fitness[i] = trial_fitness

                    if trial_fitness < func(best_solution):
                        best_solution = trial

        return best_solution