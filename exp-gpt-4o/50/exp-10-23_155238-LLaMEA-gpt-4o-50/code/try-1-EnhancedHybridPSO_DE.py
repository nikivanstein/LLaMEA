import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.9  # inertia weight
        self.f = 0.8  # DE scaling factor
        self.cr = 0.8  # DE crossover rate
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.learning_rate = np.random.uniform(0.1, 0.5, self.population_size)

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

            # Update velocities and positions using PSO with adaptive learning rate
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.learning_rate[i] = 0.9 * self.learning_rate[i] + 0.1 * np.random.rand()
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.global_best_position - self.positions[i])
                ) * self.learning_rate[i]
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i], self.lower_bound, self.upper_bound
                )

            # Apply DE mutation and crossover with adaptive scaling
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(
                    self.positions[a] + self.f * self.learning_rate[i] * (self.positions[b] - self.positions[c]),
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