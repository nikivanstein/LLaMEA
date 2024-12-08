import numpy as np

class OptimizedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(max(20, dim * 5), 100)
        self.inertia = 0.6  # Adjusted inertia weight
        self.cog_coeff = 1.5  # Adjusted cognitive coefficient
        self.soc_coeff = 1.5  # Adjusted social coefficient
        self.vel_clamp = (self.upper_bound - self.lower_bound) / 2
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.pop_size, np.inf)
        gbest_position = None
        gbest_score = np.inf
        update_freq = max(1, self.pop_size // 10)  # Frequency to update global best
        
        while self.eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, positions)
            self.eval_count += self.pop_size
            
            gbest_update = False
            for i in range(self.pop_size):
                if scores[i] < pbest_scores[i]:
                    pbest_scores[i] = scores[i]
                    pbest_positions[i] = positions[i]
                if scores[i] < gbest_score:
                    gbest_score = scores[i]
                    gbest_position = positions[i]
                    gbest_update = True

            if gbest_update or self.eval_count % update_freq == 0:
                r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
                velocities = (self.inertia * velocities +
                              self.cog_coeff * r1 * (pbest_positions - positions) +
                              self.soc_coeff * r2 * (gbest_position - positions))
                
                velocities = np.clip(velocities, -self.vel_clamp, self.vel_clamp)
                positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

        return gbest_position, gbest_score