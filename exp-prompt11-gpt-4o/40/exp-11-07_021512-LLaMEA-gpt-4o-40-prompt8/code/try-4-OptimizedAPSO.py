import numpy as np

class OptimizedAPSO:
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
        velocities = np.zeros_like(positions)  # initialize velocities to zero
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        while self.eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, positions)
            self.eval_count += self.population_size
            
            better_scores = scores < personal_best_scores
            personal_best_scores = np.where(better_scores, scores, personal_best_scores)
            personal_best_positions = np.where(better_scores[:, np.newaxis], positions, personal_best_positions)
            
            if np.min(scores) < global_best_score:
                global_best_score = np.min(scores)
                global_best_position = positions[np.argmin(scores)]

            if self.eval_count >= self.budget:
                break

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            
            cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions - positions)
            social_velocity = self.social_coeff * r2 * (global_best_position - positions)
            velocities = (self.inertia_weight * velocities) + cognitive_velocity + social_velocity
            
            np.clip(velocities, self.vel_clamp[0], self.vel_clamp[1], out=velocities)
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

        return global_best_position, global_best_score