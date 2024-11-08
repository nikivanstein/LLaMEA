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
        self.inertia_damp = 0.99  # New parameter for inertia damping

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.pop_size, np.inf)
        gbest_position = None
        gbest_score = np.inf

        while self.eval_count < self.budget:
            # Vectorized calculation of scores to improve runtime efficiency
            scores = np.apply_along_axis(func, 1, positions)
            self.eval_count += self.pop_size
            
            # Use numpy functions for updating best positions and scores
            improved_pbest_mask = scores < pbest_scores
            pbest_scores[improved_pbest_mask] = scores[improved_pbest_mask]
            pbest_positions[improved_pbest_mask] = positions[improved_pbest_mask]
            
            # Update global best without loop
            min_score_index = np.argmin(scores)
            if scores[min_score_index] < gbest_score:
                gbest_score = scores[min_score_index]
                gbest_position = positions[min_score_index]

            if self.eval_count >= self.budget:
                break

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            # Apply adaptive inertia damping
            self.inertia *= self.inertia_damp
            
            velocities = (self.inertia * velocities +
                          self.cog_coeff * r1 * (pbest_positions - positions) +
                          self.soc_coeff * r2 * (gbest_position - positions))
            
            velocities = np.clip(velocities, -self.vel_clamp, self.vel_clamp)
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

        return gbest_position, gbest_score