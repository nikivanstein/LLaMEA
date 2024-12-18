import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30  # Increased initial population size
        self.pop_size = self.initial_pop_size
        self.CR = 0.5  # Adjusted crossover rate
        self.F = 0.8  # Adjusted scaling factor
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
                dynamic_F = self.F * (0.75 + 0.25 * np.random.rand())  # More randomized dynamic F
                mutant = a + dynamic_F * (b - c) + 0.15 * (best_solution - a)  # Increased influence of best_solution
                mutant = np.clip(mutant, self.lb, self.ub)

                adaptive_CR = self.CR + 0.2 * np.random.rand() * (1 - self.evaluations / self.budget)  # More adaptive CR
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

            if self.evaluations % (self.budget // 10) == 0:  # More frequent population resizing
                self.pop_size = max(5, int(self.pop_size * 0.9))  # Adjusted reduction rate
                self.population = self.population[:self.pop_size]
                self.population_fitness = self.population_fitness[:self.pop_size]

            self.CR = 0.5 + 0.3 * np.random.rand()  # Adjusted CR range
            self.F = 0.7 + 0.25 * np.random.rand()  # Adjusted F range

        return best_solution