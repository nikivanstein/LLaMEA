import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Initial population size
        self.CR = 0.5  # Initial Crossover probability
        self.F = 0.8  # Initial Differential weight
        self.lb = -5.0  # Lower bound
        self.ub = 5.0  # Upper bound
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.population_fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        self.population_fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.pop_size
        gen_counter = 0
        restart_counter = 0

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F * (0.4 + 0.6 * (1 - (self.evaluations / self.budget)))  # Slightly more flexible mutation
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)

                crossover_mask = np.random.rand(self.dim) < (self.CR + 0.1 * np.random.rand())  # Tweaked variability
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            self.CR = 0.4 + 0.6 * np.random.rand()
            self.F = 0.6 + 0.2 * np.random.rand()

            if self.evaluations % (self.budget // 8) == 0:
                self.pop_size = max(5, int(self.pop_size * 0.85))
                self.population = self.population[:self.pop_size]
                self.population_fitness = self.population_fitness[:self.pop_size]

            if gen_counter % 110 == 0 and restart_counter < 4:
                best_idx = np.argmin(self.population_fitness)
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[0] = self.population[best_idx]
                self.population_fitness = np.full(self.pop_size, np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1
                restart_counter += 1

            gen_counter += 1

        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]