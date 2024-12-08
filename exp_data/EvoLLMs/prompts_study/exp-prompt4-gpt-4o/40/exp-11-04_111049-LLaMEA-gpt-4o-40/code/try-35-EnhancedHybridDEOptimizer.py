import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim
        self.sub_population_size = np.random.randint(5 * dim, 7 * dim)
        self.num_islands = max(1, self.population_size // self.sub_population_size)
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.success_rate = 0.3
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []
        self.dynamic_adjustment = 0.1
        self.strategy_prob = [0.6, 0.4]

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

    def mutate_rand_best_1(self, target_idx, island_idx):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        a, b = self.population[np.random.choice(indices, 2, replace=False)]
        return np.clip(self.best_solution + self.mutation_factor * (a - b), self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def migrate(self):
        for i in range(self.num_islands - 1):
            swap_idx = np.random.randint(0, self.sub_population_size)
            island_a_start = i * self.sub_population_size
            island_b_start = (i + 1) * self.sub_population_size
            self.population[[island_a_start + swap_idx, island_b_start + swap_idx]] = \
                self.population[[island_b_start + swap_idx, island_a_start + swap_idx]]

    def select_mutation_strategy(self, target_idx, island_idx):
        if np.random.rand() < self.strategy_prob[0]:
            return self.mutate_rand_best_1(target_idx, island_idx)
        else:
            return self.mutate_rand_2(target_idx, island_idx)

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
                    self.crossover_rate = 0.8 + (1 - self.success_rate) * np.random.rand()

                self.migrate()

        return self.best_solution