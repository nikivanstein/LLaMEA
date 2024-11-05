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
        self.segments = np.array_split(np.arange(self.population_size), self.num_islands)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i]

    def mutate_rand_1(self, segment_idx, target_idx):
        indices = np.delete(segment_idx, target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        return np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

    def mutate_rand_2(self, segment_idx, target_idx):
        indices = np.delete(segment_idx, target_idx)
        a, b, c, d, e = self.population[np.random.choice(indices, 5, replace=False)]
        return np.clip(a + self.mutation_factor * (b - c + d - e), self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def migrate(self):
        for i in range(self.num_islands - 1):
            swap_idx = np.random.randint(0, len(self.segments[i]))
            island_a_idx = self.segments[i][swap_idx]
            island_b_idx = self.segments[i+1][swap_idx % len(self.segments[i+1])]
            self.population[[island_a_idx, island_b_idx]] = \
                self.population[[island_b_idx, island_a_idx]]

    def select_mutation_strategy(self, target_idx, segment_idx):
        if np.random.rand() < self.strategy_prob[0]:
            return self.mutate_rand_1(segment_idx, target_idx)
        else:
            return self.mutate_rand_2(segment_idx, target_idx)

    def adjust_strategy_prob(self, success):
        self.strategy_prob[0] += self.dynamic_adjustment * (success - 0.5)
        self.strategy_prob[0] = np.clip(self.strategy_prob[0], 0.1, 0.9)
        self.strategy_prob[1] = 1.0 - self.strategy_prob[0]

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for island_idx, segment in enumerate(self.segments):
                for i in segment:
                    if evaluations >= self.budget:
                        break

                    mutant = self.select_mutation_strategy(i, segment)
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
                        self.adjust_strategy_prob(success=1)
                    else:
                        if len(self.archive) > self.population_size:
                            self.archive.pop(0)
                        self.adjust_strategy_prob(success=0)

                    self.mutation_factor = 0.5 + self.success_rate * np.random.rand()
                    self.crossover_rate = 0.7 + (1 - self.success_rate) * np.random.rand()

                self.migrate()

        return self.best_solution