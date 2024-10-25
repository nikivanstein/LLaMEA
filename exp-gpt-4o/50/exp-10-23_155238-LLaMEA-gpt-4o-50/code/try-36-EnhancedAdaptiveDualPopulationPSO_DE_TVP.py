import numpy as np

class EnhancedAdaptiveDualPopulationPSO_DE_TVP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60
        self.c1_initial = 1.5
        self.c2_initial = 2.5
        self.w_initial = 0.9
        self.w_final = 0.4
        self.f_initial = 0.7
        self.f_final = 0.3
        self.cr = 0.9
        self.positions_pso = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.2, 0.2, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions_pso)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.learning_rate = np.random.uniform(0.05, 0.3, self.population_size)
        self.memory_positions_de = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.memory_scores_de = np.full(self.population_size, np.inf)

    def __call__(self, func):
        while self.evaluations < self.budget:
            t = self.evaluations / self.budget
            c1 = self.c1_initial * (1 - t) + self.c2_initial * t
            c2 = self.c2_initial * (1 - t) + self.c1_initial * t
            w = self.w_initial * (1 - t) + self.w_final * t
            f = self.f_initial * (1 - t) + self.f_final * t

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(self.positions_pso[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions_pso[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions_pso[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.learning_rate[i] = 0.85 * self.learning_rate[i] + 0.15 * np.random.rand()
                self.velocities[i] = (
                    w * self.velocities[i]
                    + c1 * r1 * (self.personal_best_positions[i] - self.positions_pso[i])
                    + c2 * r2 * (self.global_best_position - self.positions_pso[i])
                ) * self.learning_rate[i]
                self.positions_pso[i] = np.clip(
                    self.positions_pso[i] + self.velocities[i], self.lower_bound, self.upper_bound
                )

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(
                    self.memory_positions_de[a] + f * self.learning_rate[i] * (self.memory_positions_de[b] - self.memory_positions_de[c]),
                    self.lower_bound,
                    self.upper_bound,
                )
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, self.memory_positions_de[i])
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < self.memory_scores_de[i]:
                    self.memory_positions_de[i] = trial
                    self.memory_scores_de[i] = trial_score
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial

        return self.global_best_position, self.global_best_score