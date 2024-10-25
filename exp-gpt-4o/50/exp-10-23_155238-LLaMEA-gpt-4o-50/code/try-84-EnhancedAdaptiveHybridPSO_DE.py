import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.c1 = 1.5  # Slightly reduced cognitive component for better global search
        self.c2 = 2.5  # Increased social component for improved cooperation
        self.w = 0.5  # Reduced inertia weight for more balanced exploration-exploitation
        self.f = 0.8  # Reduced mutation factor for stable convergence
        self.cr = 0.7  # Lower crossover rate to maintain diversity
        self.positions_pso = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))  # Narrowed velocity range for refined control
        self.personal_best_positions = np.copy(self.positions_pso)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.learning_rate = np.random.uniform(0.1, 0.2, self.population_size)  # Narrowed learning rate range
        self.memory_positions_de = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.memory_scores_de = np.full(self.population_size, np.inf)
        self.velocity_clamp = 1.0  # Velocity clamping for improved stability

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
                self.learning_rate[i] = 0.5 * self.learning_rate[i] + 0.5 * np.random.rand()  # Updated learning rate adaptation
                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.personal_best_positions[i] - self.positions_pso[i])
                    + self.c2 * r2 * (self.global_best_position - self.positions_pso[i])
                ) * self.learning_rate[i]
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
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