import numpy as np

class EnhancedADEOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Slightly larger population size
        self.sub_population_size = np.random.randint(3 * dim, 4 * dim)  # Adjusted sub-population size
        self.num_islands = max(1, self.population_size // self.sub_population_size)
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.7  # Adjusted mutation factor
        self.crossover_rate = 0.9  # Increased crossover rate
        self.success_rate = 0.2
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []
        self.dynamic_adjustment = 0.05
        self.strategy_prob = [0.6, 0.4]  # Adjusted strategy probabilities

    def initialize_population(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i]

    def mutate_best_1(self, target_idx, island_idx):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        best_idx = np.argmin(self.fitness[start:end]) + start
        a, b = self.population[np.random.choice(indices, 2, replace=False)]
        return np.clip(self.population[best_idx] + self.mutation_factor * (a - b), self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def migrate(self):
        migration_size = max(1, self.sub_population_size // 10)
        for i in range(self.num_islands - 1):
            swap_indices = np.random.choice(np.arange(self.sub_population_size), migration_size, replace=False)
            island_a_start = i * self.sub_population_size
            island_b_start = (i + 1) * self.sub_population_size
            for idx in swap_indices:
                idx_a = island_a_start + idx
                idx_b = island_b_start + idx
                self.population[[idx_a, idx_b]] = self.population[[idx_b, idx_a]]

    def select_mutation_strategy(self, target_idx, island_idx):
        if np.random.rand() < self.strategy_prob[0]:
            return self.mutate_best_1(target_idx, island_idx)
        else:
            return self.mutate_rand_1(target_idx, island_idx)

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for island_idx in range(self.num_islands):
                start = island_idx * self.sub_population_size
                end = start + self.sub_population_size
                for i in range(start, end):
                    if evaluations >= self.budget:
                        break

                    mutant = self.select_mutation_strategy(i, island_idx)
                    trial = self.crossover(self.population[i], mutant)

                    trial_fitness = func(trial)
                    evaluations += 1

                    if trial_fitness < self.fitness[i]:
                        if self.fitness[i] != np.inf:
                            self.archive.append(self.population[i].copy())
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                        if trial_fitness < self.best_fitness:
                            self.best_fitness = trial_fitness
                            self.best_solution = trial
                        self.success_rate = min(1.0, self.success_rate + self.dynamic_adjustment)
                        self.strategy_prob[0] = min(1.0, self.strategy_prob[0] + self.dynamic_adjustment)
                        self.strategy_prob[1] = 1.0 - self.strategy_prob[0]
                    else:
                        self.success_rate = max(0.1, self.success_rate - self.dynamic_adjustment)
                        if len(self.archive) > self.population_size:
                            self.archive.pop(0)

                    self.mutation_factor = 0.6 + self.success_rate * np.random.rand()
                    self.crossover_rate = 0.75 + (1 - self.success_rate) * np.random.rand()

                self.migrate()

        return self.best_solution