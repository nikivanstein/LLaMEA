import numpy as np

class CuckooSearchDE:
    def __init__(self, budget, dim, population_size=30, pa=0.25, alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.pa = pa
        self.alpha = alpha

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        p_best = population.copy()
        fitness = np.array([func(individual) for individual in population])
        p_best_fitness = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best = population[g_best_idx]

        for _ in range(self.budget - self.population_size):
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.alpha * (b - c), -5.0, 5.0)
                
                if np.random.rand() < self.pa:
                    j = np.random.randint(self.population_size)
                    new_population.append(mutant if func(mutant) < func(population[j]) else population[j])
                else:
                    new_population.append(population[i])

            population = np.array(new_population)
            fitness = np.array([func(individual) for individual in population])

            for i in range(self.population_size):
                if fitness[i] < p_best_fitness[i]:
                    p_best[i] = population[i]
                    p_best_fitness[i] = fitness[i]

                    if fitness[i] < fitness[g_best_idx]:
                        g_best_idx = i
                        g_best = population[i]

        return g_best