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
        restart_counter = 0  # Counter for restarts

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                dynamic_F = self.F * (0.6 + 0.4 * (1 - (self.evaluations / self.budget)))  # More flexible mutation
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)

                # Improved Crossover
                crossover_mask = np.random.rand(self.dim) < (self.CR + 0.15 * np.random.rand())  # Increased variability
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            # Stochastic adjustment of CR and F
            self.CR = 0.55 + 0.45 * np.random.rand()
            self.F = 0.55 + 0.25 * np.random.rand()

            # Adaptive population size
            if self.evaluations % (self.budget // 8) == 0:  # Adjustment for more frequent population size change
                self.pop_size = max(5, int(self.pop_size * 0.85))
                self.population = self.population[:self.pop_size]
                self.population_fitness = self.population_fitness[:self.pop_size]

            # Controlled random restart mechanism
            if gen_counter % 110 == 0 and restart_counter < 3:  # Adjusted restart frequency
                best_idx = np.argmin(self.population_fitness)
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[0] = self.population[best_idx]  # Keep the best solution
                self.population_fitness = np.full(self.pop_size, np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1
                restart_counter += 1  # Increment restart counter

            gen_counter += 1

        # Return the best solution found
        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]