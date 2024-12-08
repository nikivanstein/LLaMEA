import numpy as np

class APSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf
        self.omega = 0.5
        self.phi_p = 0.5
        self.phi_g = 0.9

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                score = func(self.positions[i])
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            r_p = np.random.uniform(0, 1, (self.population_size, self.dim))
            r_g = np.random.uniform(0, 1, (self.population_size, self.dim))
            self.velocities = (
                self.omega * self.velocities +
                self.phi_p * r_p * (self.personal_best_positions - self.positions) +
                self.phi_g * r_g * (self.global_best_position - self.positions)
            )
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adaptive parameter tuning
            success_ratio = np.sum(self.personal_best_scores < score) / self.population_size
            self.omega = max(0.4, self.omega * (1.0 if success_ratio > 0.5 else 0.9))
            self.phi_p = min(1.0, self.phi_p + (0.1 if success_ratio < 0.2 else -0.1))
            self.phi_g = max(0.5, self.phi_g - (0.1 if success_ratio > 0.7 else -0.1))

        return self.global_best_position, self.global_best_score