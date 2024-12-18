import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20  # Initial population size
        self.pop_size = self.initial_pop_size
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
        restart_counter = 0  # Counter for restarts
        prev_best_fitness = np.inf  # Track previous best fitness

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F * (0.5 + 0.5 * (1 - (self.evaluations / self.budget)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)

                crossover_mask = np.random.rand(self.dim) < (self.CR + 0.2 * np.random.rand())
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            self.CR = 0.5 + 0.5 * np.random.rand()
            self.F = 0.5 + 0.3 * np.random.rand()

            # Adaptive population size based on progress
            improvement = prev_best_fitness - np.min(self.population_fitness)
            if self.evaluations % (self.budget // 10) == 0 and improvement < 1e-5:
                self.pop_size = max(5, int(self.pop_size * 0.8))
                self.population = self.population[:self.pop_size]
                self.population_fitness = self.population_fitness[:self.pop_size]
            prev_best_fitness = np.min(self.population_fitness)

            best_idx = np.argmin(self.population_fitness)
            best_fitness = self.population_fitness[best_idx]
            if gen_counter % 115 == 0 and restart_counter < 3:
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[0] = self.population[best_idx]  
                self.population_fitness = np.full(self.pop_size, np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1
                restart_counter += 1

            # Early stopping condition if no significant improvement
            if np.abs(prev_best_fitness - best_fitness) < 1e-6:
                break

            gen_counter += 1

        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]