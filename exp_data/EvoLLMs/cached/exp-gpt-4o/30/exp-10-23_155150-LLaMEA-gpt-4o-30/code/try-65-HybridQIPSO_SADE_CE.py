import numpy as np

class HybridQIPSO_SADE_CE:
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
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.f = 0.5
        self.cr = 0.9
        self.chaotic_map = np.random.uniform(0, 1, self.dim)
    
    def chaotic_exploration(self):
        self.chaotic_map = (self.chaotic_map * (1 - self.chaotic_map)) * 4
        return self.lb + (self.ub - self.lb) * self.chaotic_map
    
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
            idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
            mutant = np.clip(self.pop[idxs[0]] + self.f * (self.pop[idxs[1]] - self.pop[idxs[2]]), self.lb, self.ub)
            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.pop[i])
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
                if self.func_evals < self.budget:
                    chaotic_pos = self.chaotic_exploration()
                    f_chaotic = func(chaotic_pos)
                    self.func_evals += 1
                    if f_chaotic < self.global_best_fitness:
                        self.global_best_fitness = f_chaotic
                        self.global_best = chaotic_pos

        return self.global_best