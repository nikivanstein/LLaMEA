import numpy as np

class AdaptiveRandomWalkEvoStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim  # Adjusted population size
        self.F = 0.6  # Slightly increased differential weight
        self.CR = 0.9  # Further increased crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_factor = 0.8  # Adaptive factor fine-tuned
        self.stagnation_threshold = 5  # Threshold for stagnation

    def initialize_population(self):
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()
        return fitness

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        np.random.shuffle(indices)
        a, b, c = indices[:3]
        return self.population[a] + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        j_rand = np.random.randint(0, self.dim)
        trial = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else target[j] for j in range(self.dim)])
        return np.clip(trial, self.bounds[0], self.bounds[1])

    def random_walk(self, solution, stddev):
        return np.clip(solution + np.random.normal(0, stddev, self.dim), self.bounds[0], self.bounds[1])

    def optimize(self, func):
        self.initialize_population()
        func_calls = 0
        stagnation_counter = 0

        while func_calls < self.budget:
            for i in range(self.population_size):
                if func_calls >= self.budget:
                    break
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                trial_fitness = func(trial)
                func_calls += 1
                if trial_fitness < func(target):
                    self.population[i] = trial
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            if stagnation_counter > self.stagnation_threshold:
                self.F *= self.adaptive_factor
                self.CR *= self.adaptive_factor
                self.F = np.clip(self.F, 0.4, 0.9)
                self.CR = np.clip(self.CR, 0.5, 1.0)

            if stagnation_counter > self.population_size // 2:
                for i in range(self.population_size):
                    if func_calls >= self.budget:
                        break
                    random_walk_solution = self.random_walk(self.population[i], 0.5 * self.F)
                    random_walk_fitness = func(random_walk_solution)
                    func_calls += 1
                    if random_walk_fitness < self.best_fitness:
                        self.best_fitness = random_walk_fitness
                        self.best_solution = random_walk_solution.copy()

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution