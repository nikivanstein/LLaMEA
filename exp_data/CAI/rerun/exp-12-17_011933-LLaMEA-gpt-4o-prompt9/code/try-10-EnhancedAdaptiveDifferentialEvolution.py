import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Adjusted initial population size
        self.CR = 0.7  # Modified Crossover probability
        self.F = 0.9  # Modified Differential weight
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
                # Hybrid Mutation with adaptive scaling
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                scale_factor = 0.6 + (0.4 * (1 - self.evaluations / self.budget))  # Adaptive scaling based on progress
                mutant = np.clip(a + scale_factor * (b - c), self.lb, self.ub)

                # Dynamic Crossover with exploration-exploitation balance
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.population_fitness[i]:
                    self.population[i] = trial
                    self.population_fitness[i] = trial_fitness

            # Adaptive population size and diversity strategy
            if self.evaluations % (self.budget // 8) == 0:
                self.pop_size = max(5, int(self.pop_size * 0.85))
                selected_indices = np.random.choice(range(self.pop_size), self.pop_size, replace=False)
                self.population = self.population[selected_indices]
                self.population_fitness = self.population_fitness[selected_indices]

            # Restart and diversity enhancement
            if gen_counter % 100 == 0:
                best_idx = np.argmin(self.population_fitness)
                self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                self.population[0] = self.population[best_idx]  
                self.population_fitness = np.full(self.pop_size, np.inf)
                self.population_fitness[0] = func(self.population[0])
                self.evaluations += 1

            gen_counter += 1

        best_idx = np.argmin(self.population_fitness)
        return self.population[best_idx]