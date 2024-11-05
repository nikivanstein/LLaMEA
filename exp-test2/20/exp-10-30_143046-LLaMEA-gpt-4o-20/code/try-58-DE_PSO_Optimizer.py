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
        self.w_decay = 0.99
        self.cr_decay = 0.995
        self.evaluations = 0

    def __call__(self, func):
        stagnation_counter = 0
        while self.evaluations < self.budget:
            self._differential_evolution_step(func)
            self._particle_swarm_step(func)
            self._adaptive_clustering(func)  # Change 1 (New function call)
            self._exploit_elitism(func)  # Change 2 (New function call)
            self._adapt_parameters()
            if stagnation_counter > 10:
                self._reinitialize_parameters()
                stagnation_counter = 0
        return self.global_best_value, self.global_best

    def _differential_evolution_step(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + np.random.uniform(0.5, 1.0) * (b - c), -5, 5)
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

    def _adaptive_clustering(self, func):  # Change 3 (New function)
        if self.evaluations < self.budget:
            centers = self.pop[np.random.choice(self.population_size, 3, replace=False)]
            for center in centers:
                cluster_point = center + np.random.normal(0, 0.5, self.dim)
                cluster_point = np.clip(cluster_point, -5, 5)
                f_cluster = func(cluster_point)
                self.evaluations += 1
                if f_cluster < self.global_best_value:
                    self.global_best_value = f_cluster
                    self.global_best = cluster_point

    def _exploit_elitism(self, func):  # Change 4 (New function)
        elite_idx = np.argmin(self.personal_best_values)
        if self.evaluations < self.budget:
            elite_candidate = self.personal_best[elite_idx] + np.random.normal(0, 0.1, self.dim)
            elite_candidate = np.clip(elite_candidate, -5, 5)
            f_elite = func(elite_candidate)
            self.evaluations += 1
            if f_elite < self.global_best_value:
                self.global_best_value = f_elite
                self.global_best = elite_candidate

    def _adapt_parameters(self):
        self.w *= self.w_decay
        self.cr *= self.cr_decay

    def _reinitialize_parameters(self):
        self.w = np.random.uniform(0.4, 0.9)
        self.cr = np.random.uniform(0.7, 1.0)
        self.pop = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best = self.pop.copy()
        self.personal_best_values.fill(np.inf)