import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w_min = 0.4
        self.w_max = 0.9
        self.k = 0.729  # constriction factor
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        iteration = 0
        while self.evaluations < self.budget:
            # Evaluate current positions
            scores = np.array([func(pos) for pos in self.positions])
            self.evaluations += self.swarm_size

            # Update personal bests
            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            # Update global best
            best_particle = np.argmin(scores)
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            # Adaptive inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * iteration / (self.budget // self.swarm_size))
            
            # Update velocities and positions
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.k * (w * self.velocities + cognitive_velocity + social_velocity)
            self.positions += self.velocities
            
            # Enforce bounds
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
            iteration += 1
            
        return self.global_best_position, self.global_best_score