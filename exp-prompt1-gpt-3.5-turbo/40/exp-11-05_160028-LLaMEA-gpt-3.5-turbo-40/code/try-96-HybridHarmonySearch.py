import numpy as np

class HybridHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5
        self.CR = 0.8

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget - len(population)):
            if np.random.rand() < 0.4:  # 40.0% chance of using Differential Evolution
                idxs = np.random.choice(np.arange(self.budget), 3, replace=False)
                mutant = population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[idxs[0]])
                trial_fitness = func(trial)
                if trial_fitness < fitness[idxs[0]]:
                    population[idxs[0]] = trial
                    fitness[idxs[0]] = trial_fitness
            else:
                new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                new_fitness = func(new_harmony)
                worst_idx = np.argmax(fitness)
                if new_fitness < fitness[worst_idx]:
                    population[worst_idx] = new_harmony
                    fitness[worst_idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]