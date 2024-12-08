import numpy as np

class AdvancedSynergisticPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60
        self.c1 = 1.6  # Further adjusted acceleration coefficient for cognitive component
        self.c2 = 2.4  # Slightly decreased acceleration coefficient for social component
        self.w = 0.65  # Increased inertia weight for enhanced exploration
        self.f = 0.8  # Increased mutation factor for greater diversity
        self.cr = 0.85  # Slightly reduced crossover rate for better DE exploration
        self.positions_pso = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))  # Broadened initial velocity range
        self.personal_best_positions = np.copy(self.positions_pso)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.learning_rate = np.random.uniform(0.1, 0.3, self.population_size)  # Changed learning rate range for dynamic adaptation
        self.memory_positions_de = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.memory_scores_de = np.full(self.population_size, np.inf)

    def __call__(self, func):
        while self.evaluations < self.budget:
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
                self.learning_rate[i] = 0.9 * self.learning_rate[i] + 0.1 * np.random.rand()  # Further refined learning rate adaptation
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best_positions[i] - self.positions_pso[i])
                    + self.c2 * r2 * (self.global_best_position - self.positions_pso[i])
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
                    self.memory_positions_de[a] + self.f * self.learning_rate[i] * (self.memory_positions_de[b] - self.memory_positions_de[c]),
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