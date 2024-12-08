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
        self.crossover_rate = 0.9
        self.success_rate = 0.2
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_adjustment = 0.05
        self.strategy_prob = [0.5, 0.5]
        self.global_learning_rate = 0.1
        self.local_learning_rate = 0.3
        self.migration_interval = 5
        self.archive = []
        self.rank_threshold = 0.45  # New parameter for stochastic ranking
        self.strategy_switch_interval = 10  # New parameter

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

    def mutate_hybrid(self, target_idx, island_idx, generation):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        if np.random.rand() < self.strategy_prob[0]:
            return np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        else:
            scale = self.global_learning_rate if generation % 2 == 0 else self.local_learning_rate
            return np.clip(a + self.mutation_factor * (b - c + scale * (self.best_solution - a)), self.lower_bound, self.upper_bound)

    def stochastic_ranking(self, trial, target, trial_fitness, target_fitness):
        if np.random.rand() < self.rank_threshold:
            return trial_fitness < target_fitness
        else:
            return np.linalg.norm(trial) < np.linalg.norm(target)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def migrate(self, generation):
        if generation % self.migration_interval == 0:
            for i in range(self.num_islands - 1):
                swap_idx = np.random.randint(0, self.sub_population_size)
                island_a_start = i * self.sub_population_size
                island_b_start = (i + 1) * self.sub_population_size
                self.population[[island_a_start + swap_idx, island_b_start + swap_idx]] = \
                    self.population[[island_b_start + swap_idx, island_a_start + swap_idx]]

    def resize_population(self):
        if self.success_rate > 0.7 and self.population_size < 20 * self.dim:
            self.population_size += 1
            new_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            self.population = np.vstack([self.population, new_individual])
            self.fitness = np.append(self.fitness, np.inf)

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        generation = 0

        while evaluations < self.budget:
            for island_idx in range(self.num_islands):
                start = island_idx * self.sub_population_size
                end = start + self.sub_population_size
                for i in range(start, end):
                    if evaluations >= self.budget:
                        break

                    mutant = self.mutate_hybrid(i, island_idx, generation)
                    trial = self.crossover(self.population[i], mutant)

                    trial_fitness = func(trial)
                    evaluations += 1

                    if self.stochastic_ranking(trial, self.population[i], trial_fitness, self.fitness[i]):
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                        if trial_fitness < self.best_fitness:
                            self.best_fitness = trial_fitness
                            self.best_solution = trial
                        self.success_rate = min(1.0, self.success_rate + self.dynamic_adjustment)
                    else:
                        self.success_rate = max(0.1, self.success_rate - self.dynamic_adjustment)

                    self.mutation_factor = 0.5 + self.local_learning_rate * np.random.rand()
                    self.crossover_rate = 0.8 + (1 - self.success_rate) * np.random.rand()

                generation += 1
                self.migrate(generation)
                if generation % self.strategy_switch_interval == 0:
                    self.strategy_prob = [np.random.rand(), 1 - np.random.rand()]
            self.resize_population()

        return self.best_solution