import numpy as np

class DynamicMutDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cr = 0.9
        self.f_min = 0.2
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def mutate(self, target_idx, population, fitnesses):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        selected = np.random.choice(candidates, 3, replace=False)
        f_dyn = np.clip(0.5 + 0.5 * (1 - fitnesses[target_idx] / max(fitnesses)), 0.2, 0.8)
        mutant = population[selected[0]] + f_dyn * (population[selected[1]] - population[selected[2]])
        return mutant

    def __call__(self, func):
        for iter_count in range(self.budget):
            for i in range(self.population_size):
                mutant = self.mutate(i, self.population, [func(ind) for ind in self.population])
                trial = self.population[i] + np.random.uniform(-0.1, 0.1, self.dim) * (mutant - self.population[i])
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
            fitnesses = np.array([func(ind) for ind in self.population])
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = np.copy(self.population[best_idx])
        return self.best_solution