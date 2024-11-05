import numpy as np

class EnhancedDynamicPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        iteration = 0
        dynamic_velocity_clamp = (self.upper_bound - self.lower_bound) * np.random.uniform(0.05, 0.15)

        while self.evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            self.evaluations += self.swarm_size

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            best_particle = np.argmin(scores)
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            w = 0.7 - ((0.6) * iteration / (self.budget // self.swarm_size))
            c1 = 1.7 - ((0.2) * iteration / (self.budget // self.swarm_size))
            c2 = 1.7 + ((0.5) * iteration / (self.budget // self.swarm_size))
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -dynamic_velocity_clamp, dynamic_velocity_clamp)
            self.positions += self.velocities

            if iteration % 10 == 0:
                diversity_factor = np.std(self.positions, axis=0)
                if np.mean(diversity_factor) < 0.1:
                    self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))

            if iteration % 20 == 0:
                differential_indices = np.random.choice(self.swarm_size, self.swarm_size // 2, replace=False)
                for idx in differential_indices:
                    partners = np.random.choice(self.swarm_size, 2, replace=False)
                    differential_vector = (self.positions[partners[0]] - self.positions[partners[1]])
                    self.positions[idx] = np.clip(self.positions[idx] + 0.5 * differential_vector, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score