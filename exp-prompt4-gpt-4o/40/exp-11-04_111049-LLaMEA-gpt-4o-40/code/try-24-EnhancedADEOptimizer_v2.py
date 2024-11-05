import numpy as np

class EnhancedADEOptimizer_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.5
        self.crossover_rate = 0.8
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []
        self.dynamic_adjustment = 0.05
        self.learning_memory = []  # Memory for adaptive learning
        self.min_population_size = 4  # Minimum population size
        self.evaluations = 0

    def initialize_population(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if np.isinf(self.fitness[i]):
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i]

    def mutate(self, target_idx):
        indices = np.arange(self.population_size)
        indices = np.delete(indices, target_idx)
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

    def adjust_population_size(self):
        if self.success_rate > 0.5 and self.population_size > self.min_population_size:
            self.population_size -= 1
        elif self.success_rate < 0.2 and self.population_size < 20 * self.dim:
            self.population_size += 1

    def __call__(self, func):
        self.initialize_population()
        self.evaluate_population(func)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    if self.fitness[i] != np.inf:
                        self.archive.append(self.population[i].copy())
                    self.learning_memory.append((self.population[i], trial))
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                else:
                    if len(self.archive) > self.population_size:
                        self.archive.pop(0)

            self.mutation_factor = 0.5 + self.dynamic_adjustment * np.random.rand()
            self.crossover_rate = 0.7 + (1 - self.dynamic_adjustment) * np.random.rand()

            self.adjust_population_size()

        return self.best_solution