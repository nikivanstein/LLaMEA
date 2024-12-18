import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.CR = 0.5
        self.F = 0.8
        self.lb = -5.0
        self.ub = 5.0
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
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F * (0.5 + 0.5 * (1 - (self.evaluations / self.budget)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)

                # Improved Crossover
                crossover_mask = np.random.rand(self.dim) < (self.CR + 0.2 * np.random.rand())
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            # Dynamic adjustment of CR and F
            self.CR = 0.5 + 0.3 * np.random.rand()
            self.F = 0.4 + 0.4 * np.random.rand()  # Adjusted range

            # Dynamic population resizing and elite retention
            if self.evaluations % (self.budget // 15) == 0:
                elite_size = max(2, int(self.pop_size * 0.1))
                best_indices = np.argsort(self.population_fitness)[:elite_size]
                elite_individuals = self.population[best_indices]
                self.pop_size = max(5, int(self.pop_size * 0.85))
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[:elite_size] = elite_individuals
                self.population_fitness = np.full(self.pop_size, np.inf)
                for j in range(elite_size):
                    self.population_fitness[j] = func(self.population[j])
                    self.evaluations += 1

            # Controlled random restart mechanism
            if gen_counter % 110 == 0 and restart_counter < 3:
                best_idx = np.argmin(self.population_fitness)
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[0] = self.population[best_idx]
                self.population_fitness = np.full(self.pop_size, np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1
                restart_counter += 1

            gen_counter += 1

        # Return the best solution found
        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]