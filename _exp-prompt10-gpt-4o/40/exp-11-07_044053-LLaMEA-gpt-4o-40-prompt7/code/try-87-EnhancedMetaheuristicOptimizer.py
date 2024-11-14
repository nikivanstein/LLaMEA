import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 6 * self.dim  # Further reduced population size for faster execution
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(*self.bounds, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.F = 0.6  # Adjusted mutation factor for initial diversity
        self.CR = 0.9  # Higher crossover rate for increased recombination
        self.evaluations = 0
        self.archive = []

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)

                # Check archive for potential new base vector
                if len(self.archive) > 0 and np.random.rand() < 0.1:
                    base_vector = self.archive[np.random.randint(len(self.archive))]
                else:
                    base_vector = self.population[a]

                # Mutation with potential archive vector
                mutant = base_vector + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    # Archive successful mutants
                    self.archive.append(trial)

                # Dynamic adjustment based on fitness improvement
                if trial_fitness < self.fitness.mean():
                    self.F = np.clip(self.F + 0.01 * (0.5 - np.random.rand()), 0.1, 0.9)
                    self.CR = np.clip(self.CR + 0.01 * (0.5 - np.random.rand()), 0.5, 1.0)

        return self.population[np.argmin(self.fitness)]

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1