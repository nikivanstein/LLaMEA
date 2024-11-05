import numpy as np

class EnhancedADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.sub_population_size = np.random.randint(4 * dim, 6 * dim)
        self.num_islands = max(1, self.population_size // self.sub_population_size)
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.8
        self.success_rate = 0.2
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []
        self.dynamic_adjustment = 0.05
        self.strategy_prob = [0.5, 0.5]

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

    def mutate_hybrid(self, target_idx, island_idx):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        selected_indices = np.random.choice(indices, 5, replace=False)
        a, b, c, d, e = self.population[selected_indices]
        if np.random.rand() < 0.7:
            return np.clip(a + self.mutation_factor * (b - c + d - e), self.lower_bound, self.upper_bound)
        else:
            return np.clip(b + self.mutation_factor * (c - a + d - e), self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def migrate(self):
        for i in range(self.num_islands - 1):
            swap_idx_a = np.random.randint(0, self.sub_population_size)
            swap_idx_b = np.random.randint(0, self.sub_population_size)
            island_a_start = i * self.sub_population_size
            island_b_start = (i + 1) * self.sub_population_size
            if self.success_rate > 0.4:  # adaptive condition for migration
                self.population[[island_a_start + swap_idx_a, island_b_start + swap_idx_b]] = \
                    self.population[[island_b_start + swap_idx_b, island_a_start + swap_idx_a]]

    def resize_population(self):
        if self.success_rate > 0.6 and self.population_size < 15 * self.dim:
            self.population_size += 1
            new_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            self.population = np.vstack([self.population, new_individual])
            self.fitness = np.append(self.fitness, np.inf)

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

                    mutant = self.mutate_hybrid(i, island_idx)
                    trial = self.crossover(self.population[i], mutant)

                    trial_fitness = func(trial)
                    evaluations += 1

                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                        if trial_fitness < self.best_fitness:
                            self.best_fitness = trial_fitness
                            self.best_solution = trial
                        self.success_rate = min(1.0, self.success_rate + self.dynamic_adjustment)
                    else:
                        self.success_rate = max(0.1, self.success_rate - self.dynamic_adjustment)

                    self.mutation_factor = 0.4 + self.success_rate * np.random.rand()
                    self.crossover_rate = 0.6 + (1 - self.success_rate) * np.random.rand()

                self.migrate()
            self.resize_population()

        return self.best_solution