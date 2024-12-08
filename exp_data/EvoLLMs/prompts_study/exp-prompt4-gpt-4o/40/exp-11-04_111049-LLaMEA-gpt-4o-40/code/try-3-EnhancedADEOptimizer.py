import numpy as np

class EnhancedADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = None
        self.fitness = None
        self.mutation_factor = 0.5  # Adjusted initial mutation factor
        self.crossover_rate = 0.8  # Adjusted crossover rate
        self.success_rate = 0.2
        self.best_solution = None
        self.best_fitness = np.inf
        self.archive = []  # Archive for storing inferior solutions

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

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(i)
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
                    self.success_rate = min(1.0, self.success_rate + 0.05)
                else:
                    self.success_rate = max(0.1, self.success_rate - 0.05)
                    if len(self.archive) > self.population_size:
                        self.archive.pop(0)

                self.mutation_factor = 0.5 + 0.3 * np.random.rand()  # Dynamic adjustment
                self.crossover_rate = 0.7 + 0.2 * np.random.rand()  # Dynamic adjustment

        return self.best_solution