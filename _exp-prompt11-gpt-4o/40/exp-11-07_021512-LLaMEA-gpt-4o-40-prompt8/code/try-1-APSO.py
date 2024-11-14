import numpy as np
from numba import njit, prange

class APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(max(20, dim * 5), 100)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.vel_clamp = (-(self.upper_bound - self.lower_bound), (self.upper_bound - self.lower_bound))
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(self.vel_clamp[0], self.vel_clamp[1], (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = np.zeros(self.dim)
        global_best_score = np.inf

        while self.eval_count < self.budget:
            scores = self.evaluate_population(func, positions)
            self.eval_count += self.population_size

            better_indices = scores < personal_best_scores
            personal_best_scores[better_indices] = scores[better_indices]
            personal_best_positions[better_indices] = positions[better_indices]

            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < global_best_score:
                global_best_score = scores[min_score_idx]
                global_best_position = positions[min_score_idx]

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_velocity = self.social_coeff * r2 * (global_best_position - positions)
            velocities = (self.inertia_weight * velocities) + cognitive_velocity + social_velocity

            velocities = np.clip(velocities, self.vel_clamp[0], self.vel_clamp[1])
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

        return global_best_position, global_best_score

    @staticmethod
    @njit(parallel=True)
    def evaluate_population(func, positions):
        n = positions.shape[0]
        scores = np.empty(n)
        for i in prange(n):
            scores[i] = func(positions[i])
        return scores