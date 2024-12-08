import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7  # inertia weight
        self.f = 0.5  # DE scaling factor
        self.cr = 0.9  # DE crossover rate
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate and update personal and global bests
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(self.positions[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # Update velocities and positions using PSO
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.global_best_position - self.positions[i])
                )
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i], self.lower_bound, self.upper_bound
                )

            # Apply DE mutation and crossover
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(
                    self.positions[a] + self.f * (self.positions[b] - self.positions[c]),
                    self.lower_bound,
                    self.upper_bound,
                )
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, self.positions[i])
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.positions[i] = trial
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

        return self.global_best_position, self.global_best_score