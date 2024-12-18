import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
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
            ranked_idx = np.argsort(self.population_fitness)
            self.CR = 0.5 + 0.2 * (1 - np.arange(self.pop_size) / self.pop_size)  # Dynamic CR based on rank
            for i in range(self.pop_size):
                # Mutation with best individual influence
                a, b, c = self.population[np.random.choice(ranked_idx[:self.pop_size//2], 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR[i]
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            # Dynamic F adjustment based on performance
            f_diff = np.max(self.population_fitness) - np.min(self.population_fitness)
            self.F = 0.5 + 0.3 * (f_diff / (f_diff + 1e-9))

            # Restart mechanism with elitism
            if self.evaluations + self.pop_size >= self.budget:
                break
            if gen_counter % 50 == 0:  # Increased restart frequency
                best_idx = np.argmin(self.population_fitness)
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[0] = self.population[best_idx]
                self.population_fitness.fill(np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1

            gen_counter += 1

        # Return the best solution found
        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]