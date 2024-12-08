import numpy as np

class EnhancedIslandMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 5 * self.dim  # Reduced population size for quicker processing
        self.island_count = 3  # Number of islands for diversity
        self.bounds = (-5.0, 5.0)
        self.population = [np.random.uniform(*self.bounds, (self.pop_size // self.island_count, self.dim)) for _ in range(self.island_count)]
        self.fitness = [np.full(self.pop_size // self.island_count, np.inf) for _ in range(self.island_count)]
        self.F = 0.5  # Mutation factor tuned for better convergence
        self.CR = 0.85  # Crossover rate tweaked for recombination
        self.evaluations = 0
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for island in range(self.island_count):
                for i in range(self.pop_size // self.island_count):
                    if self.evaluations >= self.budget:
                        break
                    # Select different indices for mutation
                    candidates = list(range(self.pop_size // self.island_count))
                    candidates.remove(i)
                    a, b, c = np.random.choice(candidates, 3, replace=False)

                    # Mutation and recombination
                    mutant = self.population[island][a] + self.F * (self.population[island][b] - self.population[island][c])
                    mutant = np.clip(mutant, *self.bounds)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[island][i])

                    trial_fitness = func(trial)
                    self.evaluations += 1

                    # Replace if improved
                    if trial_fitness < self.fitness[island][i]:
                        self.population[island][i] = trial
                        self.fitness[island][i] = trial_fitness
                        if trial_fitness < self.best_fitness:
                            self.best_fitness = trial_fitness
                            self.best_solution = trial

                # Perform local search based on probability
                if np.random.rand() < 0.2:
                    self.local_search(island, func)

        return self.best_solution

    def evaluate_population(self, func):
        for island in range(self.island_count):
            for i in range(self.pop_size // self.island_count):
                if self.evaluations >= self.budget:
                    break
                self.fitness[island][i] = func(self.population[island][i])
                self.evaluations += 1
                if self.fitness[island][i] < self.best_fitness:
                    self.best_fitness = self.fitness[island][i]
                    self.best_solution = self.population[island][i]

    def local_search(self, island, func):
        idx = np.argmin(self.fitness[island])
        candidate = self.population[island][idx].copy()
        perturbation = (np.random.rand(self.dim) - 0.5) * 0.1
        candidate = np.clip(candidate + perturbation, *self.bounds)
        candidate_fitness = func(candidate)
        self.evaluations += 1
        if candidate_fitness < self.fitness[island][idx]:
            self.population[island][idx] = candidate
            self.fitness[island][idx] = candidate_fitness
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = candidate