import numpy as np

class Enhanced_AOWO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(30, self.budget // 5)
        self.population_size = self.initial_population_size
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.dynamic_scale = 0.5
        self.F = 0.5  # Differential evolution factor
        self.CR = 0.9  # Crossover rate for DE

    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def reduce_dimensionality(self, solution, factor):
        mask = np.random.rand(self.dim) < factor
        reduced_solution = solution.copy()
        reduced_solution[mask] = self.best_solution[mask] if self.best_solution is not None else 0
        return reduced_solution

    def differential_evolution(self, target_idx):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.whales[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.whales[target_idx])
        return trial

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.whales[min_fitness_idx].copy()

            reduction_factor = 1 - (evaluations / self.budget)

            # Adaptive population size
            self.population_size = max(5, int(self.initial_population_size * (1 - evaluations / self.budget)))
            self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                self.dynamic_scale = 0.5 * (1 + np.cos(np.pi * evaluations / self.budget))

                if np.random.rand() < 0.5:
                    trial = self.differential_evolution(i)
                else:
                    opp_solution = self.oppositional_solution(self.whales[i])
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    trial = opp_solution - A * D * self.dynamic_scale

                trial = self.reduce_dimensionality(trial, reduction_factor)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    self.whales[i] = trial
                    fitness[i] = trial_fitness

        return self.best_solution, self.best_fitness