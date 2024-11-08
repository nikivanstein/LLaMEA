import numpy as np

class AdaptiveMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 5 * self.dim  # Reduced population size
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.F = 0.5  # Initial lower mutation factor
        self.CR = 0.8  # Adjusted crossover rate
        self.evaluations = 0
        self.archive = []
        self.archive_limit = 50  # Limit archive size

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                a, b = np.random.choice(self.pop_size, 2, replace=False)

                # Select base vector and dynamic mutation
                if len(self.archive) > 0 and np.random.rand() < 0.2:
                    base_vector = self.archive[np.random.randint(len(self.archive))]
                else:
                    base_vector = self.population[i]

                mutant = base_vector + self.adaptive_mutation() * (self.population[a] - self.population[b])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    # Archive successful mutants with limit
                    if len(self.archive) < self.archive_limit:
                        self.archive.append(trial)
                    else:
                        self.archive[np.random.randint(self.archive_limit)] = trial

                # Adjust mutation strategy based on fitness
                if trial_fitness < self.fitness.mean():
                    self.F = np.clip(self.F + 0.01 * (0.5 - np.random.rand()), 0.2, 0.8)
                    self.CR = np.clip(self.CR + 0.01 * (0.6 - np.random.rand()), 0.4, 1.0)

        return self.population[np.argmin(self.fitness)]

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

    def adaptive_mutation(self):
        return 0.5 + 0.3 * np.random.rand()