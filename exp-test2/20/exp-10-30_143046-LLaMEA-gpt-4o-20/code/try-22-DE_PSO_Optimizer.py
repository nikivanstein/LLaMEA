import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.pop = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best = self.pop.copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.cr = 0.9  
        self.f = 0.8   
        self.w = 0.5   
        self.c1 = 1.5  
        self.c2 = 1.5  
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            self._dynamic_strategy_adjustment(func)
            self._differential_evolution_step(func)
            self._particle_swarm_step(func)
        return self.global_best_value, self.global_best

    def _dynamic_strategy_adjustment(self, func):
        self.f = 0.5 + 0.5 * np.sin(0.5 * np.pi * self.evaluations / self.budget)  # Dynamic F adaptation
        if self.evaluations % (self.budget // 10) == 0:  # Periodic chaos-based local search
            self._chaos_local_search(func)

    def _differential_evolution_step(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), -5, 5)  # Updated F
            trial = np.where(np.random.rand(self.dim) < self.cr, mutant, self.pop[i])
            f_trial = func(trial)
            self.evaluations += 1
            if f_trial < self.personal_best_values[i]:
                self.personal_best_values[i] = f_trial
                self.personal_best[i] = trial
            if f_trial < self.global_best_value:
                self.global_best_value = f_trial
                self.global_best = trial

    def _particle_swarm_step(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocity[i] = (self.w * self.velocity[i] +
                                self.c1 * r1 * (self.personal_best[i] - self.pop[i]) +
                                self.c2 * r2 * (self.global_best - self.pop[i]))
            self.pop[i] = np.clip(self.pop[i] + self.velocity[i], -5, 5)
            f_value = func(self.pop[i])
            self.evaluations += 1
            if f_value < self.personal_best_values[i]:
                self.personal_best_values[i] = f_value
                self.personal_best[i] = self.pop[i]
            if f_value < self.global_best_value:
                self.global_best_value = f_value
                self.global_best = self.pop[i]

    def _chaos_local_search(self, func):
        local_candidate = self.global_best + np.random.uniform(-0.05, 0.05, self.dim) * np.sin(self.evaluations)
        local_candidate = np.clip(local_candidate, -5, 5)
        f_local = func(local_candidate)
        self.evaluations += 1
        if f_local < self.global_best_value:
            self.global_best_value = f_local
            self.global_best = local_candidate