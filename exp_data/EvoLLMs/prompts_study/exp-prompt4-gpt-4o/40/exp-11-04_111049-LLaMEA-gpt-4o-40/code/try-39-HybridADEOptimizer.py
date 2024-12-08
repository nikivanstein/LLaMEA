import numpy as np

class HybridADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Increased population size
        self.sub_population_size = np.random.randint(5 * dim, 7 * dim)  # Adjusted range
        self.num_islands = max(1, self.population_size // self.sub_population_size)
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9  # Increased initial crossover rate
        self.success_rate = 0.3  # Adjust success rate
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []
        self.dynamic_adjustment = 0.1  # Increased adjustment rate
        self.strategy_prob = [0.4, 0.6]  # Changed strategy probabilities
        self.adaptive_archive_size = 20  # New parameter for archive control
        self.niching_factor = 0.1  # New parameter for niche creation

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

    def mutate_rand_1(self, target_idx, island_idx):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        selected = np.random.choice(indices, 3, replace=False)
        a, b, c = self.population[selected]
        return np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

    def mutate_rand_2(self, target_idx, island_idx):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        selected = np.random.choice(indices, 5, replace=False)
        a, b, c, d, e = self.population[selected]
        return np.clip(a + self.mutation_factor * (b - c + d - e), self.lower_bound, self.upper_bound)

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

    def niche_creation(self):
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                if np.linalg.norm(self.population[i] - self.population[j]) < self.niching_factor:
                    perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                    self.population[j] = np.clip(self.population[j] + perturbation, self.lower_bound, self.upper_bound)

    def select_mutation_strategy(self, target_idx, island_idx):
        if np.random.rand() < self.strategy_prob[0]:
            return self.mutate_rand_1(target_idx, island_idx)
        else:
            return self.mutate_rand_2(target_idx, island_idx)

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            self.niche_creation()  # Apply niche creation
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
                        if len(self.archive) > self.adaptive_archive_size:  # Control archive size
                            self.archive.pop(0)
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

                    self.mutation_factor = 0.4 + self.success_rate * np.random.rand()
                    self.crossover_rate = 0.6 + (1 - self.success_rate) * np.random.rand()

                self.migrate()

        return self.best_solution