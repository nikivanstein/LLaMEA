import numpy as np

class OptimizedEnhancedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(max(20, dim * 5), 100)
        self.inertia = 0.7
        self.cog_coeff = 1.4
        self.soc_coeff = 1.4
        self.vel_clamp = self.upper_bound - self.lower_bound
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.pop_size, np.inf)
        gbest_position = None
        gbest_score = np.inf

        while self.eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, positions)
            self.eval_count += self.pop_size

            update_mask = scores < pbest_scores
            pbest_scores[update_mask] = scores[update_mask]
            pbest_positions[update_mask] = positions[update_mask]

            min_idx = np.argmin(scores)
            if scores[min_idx] < gbest_score:
                gbest_score = scores[min_idx]
                gbest_position = positions[min_idx]

            if self.eval_count >= self.budget:
                break

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            cognitive_component = self.cog_coeff * r1 * (pbest_positions - positions)
            social_component = self.soc_coeff * r2 * (gbest_position - positions)
            velocities = self.inertia * velocities + cognitive_component + social_component
            velocities = np.clip(velocities, -self.vel_clamp, self.vel_clamp)
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

        return gbest_position, gbest_score