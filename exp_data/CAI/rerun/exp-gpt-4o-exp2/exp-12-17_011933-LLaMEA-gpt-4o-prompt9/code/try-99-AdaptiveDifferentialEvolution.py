import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25
        self.CR = 0.7
        self.F = 0.9
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.population_fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        self.population_fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.pop_size
        best_idx = np.argmin(self.population_fitness)
        best_solution = self.population[best_idx]
        best_fitness = self.population_fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F * (0.65 + 0.35 * (1 - (self.evaluations / self.budget) ** 2))  # Changed exponent to 2 for dynamic_F
                mutant = a + dynamic_F * (b - c) + 0.1 * (best_solution - a)
                mutant = np.clip(mutant, self.lb, self.ub)

                adaptive_CR = self.CR + 0.15 * np.random.rand() * (1 - self.evaluations / self.budget) 
                crossover_mask = np.random.rand(self.dim) < adaptive_CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            current_best_idx = np.argmin(self.population_fitness)
            current_best_fitness = self.population_fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_solution = self.population[current_best_idx]
                best_fitness = current_best_fitness

            self.CR = 0.6 + 0.4 * np.random.rand()
            self.F = 0.6 + 0.2 * np.random.rand()

            if self.evaluations % (self.budget // 8) == 0:
                self.pop_size = max(5, int(self.pop_size * 0.85))
                self.population = self.population[:self.pop_size]
                self.population_fitness = self.population_fitness[:self.pop_size]

        return best_solution