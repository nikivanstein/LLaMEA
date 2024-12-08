import numpy as np

class EnhancedADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.sub_population_size = 5 * dim
        self.num_islands = 2
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.8
        self.success_rate = 0.2
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []
        self.dynamic_adjustment = 0.05
        self.crowding_distance = np.zeros(self.population_size)

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

    def mutate(self, target_idx, island_idx):
        start = island_idx * self.sub_population_size
        end = start + self.sub_population_size
        indices = np.arange(start, end)
        indices = np.delete(indices, target_idx - start)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        if self.archive:
            d = self.archive[np.random.randint(len(self.archive))]
            mutant = np.clip(a + self.mutation_factor * (b - c + d - a), self.lower_bound, self.upper_bound)
        else:
            mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def calculate_crowding_distance(self):
        for i in range(self.population_size):
            dist = 0
            for j in range(self.population_size):
                if i != j:
                    dist += np.linalg.norm(self.population[i] - self.population[j])
            self.crowding_distance[i] = dist

    def migrate(self):
        self.calculate_crowding_distance()
        for i in range(self.num_islands - 1):
            swap_idx = np.argmax(self.crowding_distance[i * self.sub_population_size:(i + 1) * self.sub_population_size])
            island_a_start = i * self.sub_population_size
            island_b_start = (i + 1) * self.sub_population_size
            self.population[[island_a_start + swap_idx, island_b_start + swap_idx]] = \
                self.population[[island_b_start + swap_idx, island_a_start + swap_idx]]

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

                    mutant = self.mutate(i, island_idx)
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
                    else:
                        self.success_rate = max(0.1, self.success_rate - self.dynamic_adjustment)
                        if len(self.archive) > self.population_size:
                            self.archive.pop(0)

                    self.mutation_factor = 0.5 + self.success_rate * np.random.rand()
                    self.crossover_rate = 0.7 + (1 - self.success_rate) * np.random.rand()

                self.migrate()

        return self.best_solution