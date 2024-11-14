import numpy as np

class OptimizedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.inertia_start = 0.9
        self.inertia_end = 0.4
        self.cognitive_const = 1.49445
        self.social_const = 1.49445
        self.particles = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = np.copy(self.particles[0])
        self.global_best_score = np.inf
        self.evaluation_count = 0

    def update_inertia(self, iteration, max_iterations):
        return self.inertia_start - (self.inertia_start - self.inertia_end) * (iteration / max_iterations)

    def __call__(self, func):
        iterations = self.budget // self.pop_size
        for iteration in range(iterations):
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluation_count += self.pop_size

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.particles[better_mask]

            min_score_index = np.argmin(scores)
            if scores[min_score_index] < self.global_best_score:
                self.global_best_score = scores[min_score_index]
                self.global_best_position = self.particles[min_score_index].copy()

            inertia_weight = self.update_inertia(iteration, iterations)
            random_coeffs = np.random.rand(self.pop_size, self.dim, 2)

            cognitive_component = self.cognitive_const * random_coeffs[:, :, 0] * (self.personal_best_positions - self.particles)
            social_component = self.social_const * random_coeffs[:, :, 1] * (self.global_best_position - self.particles)

            self.velocities = inertia_weight * self.velocities + cognitive_component + social_component
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

        return self.global_best_position, self.global_best_score