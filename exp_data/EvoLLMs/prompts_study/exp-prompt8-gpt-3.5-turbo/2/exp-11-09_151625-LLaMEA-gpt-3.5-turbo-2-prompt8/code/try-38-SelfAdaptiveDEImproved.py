import numpy as np

class SelfAdaptiveDEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9
        self.F = 0.5
        self.mutation_prob = 1.0

    def __call__(self, func):
        def mutation(target, population, diversity, fitness, iteration):
            self.mutation_prob = np.clip(1.0 / (1.0 + diversity), 0.1, 0.9)
            return population[np.random.choice(range(len(population)), 2, replace=False)]

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(10)]
        fitness = [func(ind) for ind in population]
        diversity = np.std(population)

        for _ in range(self.budget):
            new_population = []
            for idx, target in enumerate(population):
                a, b = mutation(target, population, diversity, fitness, _)
                trial = target + self.F * (a - b)
                for i in range(self.dim):
                    if np.random.rand() > self.CR:
                        trial[i] = target[i]
                f_trial = func(trial)
                if f_trial < fitness[idx]:
                    population[idx] = trial
                    fitness[idx] = f_trial
                    if f_trial < best_fitness:
                        best_solution = trial
                        best_fitness = f_trial
            diversity = np.std(population)

        return best_solution