import numpy as np
from joblib import Parallel, delayed

class OptimizedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(max(20, dim * 5), 100)
        self.inertia = 0.9  # Adaptive inertia start
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
            scores = Parallel(n_jobs=-1)(delayed(func)(positions[i]) for i in range(self.pop_size))
            self.eval_count += self.pop_size

            scores = np.array(scores)  # Vectorized post-processing
            better_mask = scores < pbest_scores
            pbest_scores[better_mask] = scores[better_mask]
            pbest_positions[better_mask] = positions[better_mask]

            if np.min(scores) < gbest_score:
                min_idx = np.argmin(scores)
                gbest_score = scores[min_idx]
                gbest_position = positions[min_idx]

            if self.eval_count >= self.budget:
                break

            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.cog_coeff * r1 * (pbest_positions - positions) +
                          self.soc_coeff * r2 * (gbest_position - positions))
            
            velocities = np.clip(velocities, -self.vel_clamp, self.vel_clamp)
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            self.inertia = 0.4 + 0.5 * ((self.budget - self.eval_count) / self.budget)  # Adaptive inertia update

        return gbest_position, gbest_score