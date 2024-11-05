import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1_initial = 2.0  # Cognitive component initial
        self.c2_initial = 2.0  # Social component initial
        self.c1 = self.c1_initial
        self.c2 = self.c2_initial
        self.w = 0.5   # Inertia weight
        self.f = 0.8   # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm = None
        self.personal_best = None
        self.global_best = None
        self.velocities = None
        self.func_evals = 0

    def initialize(self):
        self.swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.swarm.copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_score = np.inf

    def update_velocities_and_positions(self):
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive = self.c1 * r1 * (self.personal_best[i] - self.swarm[i])
            social = self.c2 * r2 * (self.global_best - self.swarm[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            self.swarm[i] += self.velocities[i]
            self.swarm[i] = np.clip(self.swarm[i], self.lower_bound, self.upper_bound)

    def differential_evolution(self):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.swarm[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
            crossover = np.random.rand(self.dim) < self.cr
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            trial = np.where(crossover, mutant, self.swarm[i])
            yield i, trial

    def __call__(self, func):
        self.initialize()
        
        while self.func_evals < self.budget:
            # Adaptive update for c1 and c2 based on function evaluations
            progress = self.func_evals / self.budget
            self.c1 = self.c1_initial * (1 - progress) + 1.0 * progress
            self.c2 = self.c2_initial * progress + 1.0 * (1 - progress)

            diversity = np.std(self.swarm, axis=0).mean()
            if diversity < 0.1:  # Dynamic switch based on population diversity
                for i, trial in self.differential_evolution():
                    if self.func_evals >= self.budget:
                        break
                    trial_score = func(trial)
                    self.func_evals += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best[i] = trial
                        self.personal_best_scores[i] = trial_score
                        if trial_score < self.global_best_score:
                            self.global_best = trial
                            self.global_best_score = trial_score

            self.update_velocities_and_positions()

            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                current_score = func(self.swarm[i])
                self.func_evals += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best[i] = self.swarm[i]
                    self.personal_best_scores[i] = current_score
                    if current_score < self.global_best_score:
                        self.global_best = self.swarm[i]
                        self.global_best_score = current_score

        return self.global_best