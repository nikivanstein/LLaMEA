import numpy as np

class APSOCovarianceOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5 + int(3 * np.log(self.dim))
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0

    def __call__(self, func):
        inertia_weight = 0.7
        cognitive_coefficient = 1.5
        social_coefficient = 1.5
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                score = func(self.positions[i])
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.positions[i])
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.positions[i])

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = cognitive_coefficient * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = social_coefficient * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity

                # Covariance adaptation
                cov_matrix = np.cov(self.positions.T)
                self.positions[i] += np.dot(np.linalg.cholesky(cov_matrix), self.velocities[i])
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score