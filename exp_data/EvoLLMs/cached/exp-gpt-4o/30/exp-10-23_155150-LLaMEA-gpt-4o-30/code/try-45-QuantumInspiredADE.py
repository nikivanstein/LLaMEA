import numpy as np

class QuantumInspiredADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.personal_best = np.copy(self.pop)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.func_evals = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def mutate(self, idx):
        indices = [index for index in range(self.pop_size) if index != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        mutant = np.clip(self.pop[r1] + self.F * (self.pop[r2] - self.pop[r3]), self.lb, self.ub)
        return mutant

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dim) < self.CR
        trial = np.where(mask, mutant, target)
        return trial

    def update(self, idx, func):
        target = self.pop[idx]
        mutant = self.mutate(idx)
        trial = self.crossover(target, mutant)
        f_trial = func(trial)
        self.func_evals += 1
        if f_trial < self.personal_best_fitness[idx]:
            self.personal_best_fitness[idx] = f_trial
            self.personal_best[idx] = trial
        if f_trial < self.global_best_fitness:
            self.global_best_fitness = f_trial
            self.global_best = trial
        if f_trial < func(target):
            self.pop[idx] = trial

    def __call__(self, func):
        for i in range(self.pop_size):
            f_value = func(self.pop[i])
            self.func_evals += 1
            self.personal_best_fitness[i] = f_value
            self.personal_best[i] = self.pop[i]
            if f_value < self.global_best_fitness:
                self.global_best_fitness = f_value
                self.global_best = self.pop[i]

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.global_best
                self.update(i, func)

        return self.global_best