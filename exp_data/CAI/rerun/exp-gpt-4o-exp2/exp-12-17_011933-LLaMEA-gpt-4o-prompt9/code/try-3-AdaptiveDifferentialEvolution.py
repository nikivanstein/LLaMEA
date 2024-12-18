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

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                # Crossover with enhanced diversity preservation
                crossover_mask = np.random.rand(self.dim) < (self.CR + 0.1 * np.random.rand())
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            # Stochastic adjustment of CR and F
            self.CR = 0.5 + 0.5 * np.random.rand()
            self.F = 0.5 + 0.3 * np.random.rand()

            # Dynamic population resizing
            if gen_counter % 50 == 0 and self.evaluations + self.pop_size < self.budget:
                if np.var(self.population_fitness) < 0.01:
                    self.pop_size = max(10, self.pop_size // 2)
                    self.population = self.population[:self.pop_size]
                    self.population_fitness = self.population_fitness[:self.pop_size]

            if gen_counter % 100 == 0:
                best_idx = np.argmin(self.population_fitness)
                self.population = np.random.uniform(self.lb, self.ub, (self.initial_pop_size, self.dim))
                self.population[0] = self.population[best_idx]
                self.population_fitness.fill(np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1
                self.pop_size = self.initial_pop_size

            gen_counter += 1

        # Return the best solution found
        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]