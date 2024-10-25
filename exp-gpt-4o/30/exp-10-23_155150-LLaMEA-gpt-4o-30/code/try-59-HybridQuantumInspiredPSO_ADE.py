import numpy as np

class HybridQuantumInspiredPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 * dim
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, dim))
        self.personal_best = np.copy(self.pop)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.func_evals = 0
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 2.0
        self.inertia_weight = 0.8
        self.cognitive_weight = 1.7
        self.social_weight = 1.3
        self.f = 0.8
        self.cr = 0.9

    def update_particle(self, i, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                              self.cognitive_weight * r1 * (self.personal_best[i] - self.pop[i]) +
                              self.social_weight * r2 * (self.global_best - self.pop[i]))
        self.pop[i] = np.clip(self.pop[i] + self.velocities[i], self.lb, self.ub)
        f_value = func(self.pop[i])
        self.func_evals += 1
        if f_value < self.personal_best_fitness[i]:
            self.personal_best_fitness[i] = f_value
            self.personal_best[i] = self.pop[i]
        if f_value < self.global_best_fitness:
            self.global_best_fitness = f_value
            self.global_best = self.pop[i]

    def differential_evolution(self, func):
        for i in range(self.pop_size):
            if self.func_evals >= self.budget:
                return

            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), self.lb, self.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.pop[i])
            f_trial = func(trial)
            self.func_evals += 1

            if f_trial < self.personal_best_fitness[i]:
                self.pop[i] = trial
                self.personal_best_fitness[i] = f_trial
                self.personal_best[i] = trial
                if f_trial < self.global_best_fitness:
                    self.global_best_fitness = f_trial
                    self.global_best = trial

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
                self.update_particle(i, func)

            if self.func_evals < self.budget:
                self.differential_evolution(func)

        return self.global_best