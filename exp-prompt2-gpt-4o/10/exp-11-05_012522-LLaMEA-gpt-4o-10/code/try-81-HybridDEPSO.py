import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 20
        self.F = 0.8
        self.CR = 0.9
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)
        self.populations = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_scores = None

    def initialize(self):
        self.population_size = self.initial_population_size
        self.populations = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, (self.population_size, self.dim))
        self.personal_best = np.copy(self.populations)
        self.personal_best_scores = np.full(self.population_size, np.inf)
    
    def differential_evolution(self, func, current_eval):
        for i in range(self.population_size):
            if current_eval >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.populations[np.random.choice(indices, 3, replace=False)]
            F_adaptive = self.F * (1 + 0.1 * np.sin(current_eval / self.budget * np.pi))
            mutant = np.clip(a + F_adaptive * (b - c), self.lower_bound, self.upper_bound)
            CR_adaptive = self.CR * (1 + 0.05 * np.cos(current_eval / self.budget * np.pi))
            trial = np.where(np.random.rand(self.dim) < CR_adaptive, mutant, self.populations[i])
            trial_score = func(trial)
            current_eval += 1
            if trial_score < self.personal_best_scores[i]:
                self.populations[i] = trial
                self.personal_best_scores[i] = trial_score
                self.personal_best[i] = trial
        return current_eval
    
    def particle_swarm_optimization(self, func, current_eval):
        global_best_idx = np.argmin(self.personal_best_scores)
        global_best = self.personal_best[global_best_idx]
        
        for i in range(self.population_size):
            if current_eval >= self.budget:
                break
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = (self.w * self.velocities[i] + 
                                  self.c1 * r1 * (self.personal_best[i] - self.populations[i]) + 
                                  self.c2 * r2 * (global_best - self.populations[i]))
            self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
            self.populations[i] = np.clip(self.populations[i] + self.velocities[i], self.lower_bound, self.upper_bound)
            score = func(self.populations[i])
            current_eval += 1
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best[i] = np.copy(self.populations[i])
        return current_eval

    def resize_population(self, current_eval):
        if current_eval >= self.budget / 2:
            self.population_size = max(10, self.population_size // 2)
            self.populations = self.populations[:self.population_size]
            self.velocities = self.velocities[:self.population_size]
            self.personal_best = self.personal_best[:self.population_size]
            self.personal_best_scores = self.personal_best_scores[:self.population_size]
    
    def __call__(self, func):
        self.initialize()
        current_eval = 0

        while current_eval < self.budget:
            self.resize_population(current_eval)
            current_eval = self.differential_evolution(func, current_eval)
            if current_eval < self.budget:
                current_eval = self.particle_swarm_optimization(func, current_eval)
        
        best_idx = np.argmin(self.personal_best_scores)
        return self.personal_best[best_idx], self.personal_best_scores[best_idx]