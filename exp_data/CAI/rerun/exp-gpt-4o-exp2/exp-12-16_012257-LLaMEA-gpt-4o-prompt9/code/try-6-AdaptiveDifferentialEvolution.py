import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_individual = None
        self.best_value = float('inf')
        self.history = []

    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, individual, func):
        step_size = 0.1
        for _ in range(5):  # Perform a few local search steps
            candidate = individual + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, self.lb, self.ub)
            if func(candidate) < func(individual):
                individual = candidate
        return individual

    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        if trial_value < self.best_value:
            self.best_value = trial_value
            self.best_individual = trial
        if trial_value < func(self.population[target_idx]):
            self.population[target_idx] = trial

    def adaptive_parameters(self):
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.7, 1.0)

    def __call__(self, func):
        self.initialize_population()
        evals = 0
        while evals < self.budget:
            self.adaptive_parameters()  # Adjust parameters adaptively
            for idx in range(self.population_size):
                if evals >= self.budget:
                    break
                target = self.population[idx]
                mutant = self.mutate(idx)
                trial = self.crossover(target, mutant)
                trial = self.local_search(trial, func)  # Apply local search
                self.select(idx, trial, func)
                evals += 1
        return self.best_individual, self.best_value