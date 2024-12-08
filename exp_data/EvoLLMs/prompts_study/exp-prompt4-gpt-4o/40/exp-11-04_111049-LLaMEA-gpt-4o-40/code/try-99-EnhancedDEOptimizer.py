import numpy as np

class EnhancedDEOptimizer:
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
        self.mutation_factor = 0.7  # Modified mutation factor
        self.crossover_rate = 0.85  # Adjusted crossover rate
        self.success_rate = 0.2
        self.best_solution = None
        self.best_fitness = np.inf
        self.dynamic_adjustment = 0.05
        self.strategy_prob = [0.5, 0.5]
        self.global_learning_rate = 0.1
        self.local_learning_rate = 0.3
        self.migration_interval = 5
        self.archive = []
        self.diversity_factor = 0.2  # New parameter for diversity enhancement
    
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
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        if np.random.rand() < 0.5:
            return np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        else:
            diversity_adjustment = self.diversity_factor * np.random.randn(self.dim)  # New diversity adjustment
            return np.clip(a + self.mutation_factor * (b - c + self.global_learning_rate * (self.best_solution - a)) + diversity_adjustment, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        weights = np.random.rand(self.dim)
        crossover_mask = weights < (self.crossover_rate * np.abs(mutant - target))  # Modified crossover strategy
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

                generation += 1
                self.migrate(generation)
            self.resize_population()

        return self.best_solution