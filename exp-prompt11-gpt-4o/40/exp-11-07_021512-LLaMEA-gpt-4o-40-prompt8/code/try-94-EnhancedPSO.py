import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(max(20, dim * 5), 100)
        self.inertia = 0.7
        self.cog_coeff = 1.4
        self.soc_coeff = 1.4
        self.vel_clamp = (self.upper_bound - self.lower_bound) / 2
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.zeros_like(positions)
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.pop_size, np.inf)

        scores = func(positions)
        self.eval_count += self.pop_size
        
        gbest_idx = np.argmin(scores)
        gbest_score = scores[gbest_idx]
        gbest_position = positions[gbest_idx].copy()

        while self.eval_count < self.budget:
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)
            
            velocities *= self.inertia
            velocities += self.cog_coeff * r1 * (pbest_positions - positions)
            velocities += self.soc_coeff * r2 * (gbest_position - positions)
            np.clip(velocities, -self.vel_clamp, self.vel_clamp, out=velocities)
            
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

            scores = func(positions)
            self.eval_count += self.pop_size

            improvement_mask = scores < pbest_scores
            np.copyto(pbest_scores, scores, where=improvement_mask)
            np.copyto(pbest_positions, positions, where=improvement_mask[:, np.newaxis])

            current_gbest_idx = np.argmin(scores)
            if scores[current_gbest_idx] < gbest_score:
                gbest_score = scores[current_gbest_idx]
                gbest_position = positions[current_gbest_idx].copy()

        return gbest_position, gbest_score