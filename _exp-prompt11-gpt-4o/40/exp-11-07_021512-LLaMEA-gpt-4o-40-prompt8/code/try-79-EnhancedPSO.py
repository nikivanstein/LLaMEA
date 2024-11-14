import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(max(20, dim * 5), 100)
        self.inertia = 0.7  # Adjusted inertia weight for better exploration
        self.cog_coeff = 1.4  # Adjusted cognitive coefficient
        self.soc_coeff = 1.4  # Adjusted social coefficient
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
        score_updates = 0

        # Evaluate initial population
        scores = np.apply_along_axis(func, 1, positions)
        self.eval_count += self.pop_size
        
        for i in range(self.pop_size):
            if scores[i] < pbest_scores[i]:
                pbest_scores[i] = scores[i]
                pbest_positions[i] = positions[i]
            if scores[i] < gbest_score:
                gbest_score = scores[i]
                gbest_position = positions[i]
                score_updates += 1

        while self.eval_count < self.budget:
            # Asynchronous evaluation and update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cog_coeff * r1 * (pbest_positions[i] - positions[i]) +
                                 self.soc_coeff * r2 * (gbest_position - positions[i]))
                velocities[i] = np.clip(velocities[i], -self.vel_clamp, self.vel_clamp)
                positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                score = func(positions[i])
                self.eval_count += 1
                
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i]
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i]
                    score_updates += 1
                    
                # Stop if budget is exhausted
                if self.eval_count >= self.budget:
                    break

        return gbest_position, gbest_score