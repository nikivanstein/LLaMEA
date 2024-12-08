import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1_base, self.c2_base = 2.5, 2.5
        self.w_min, self.w_max = 0.4, 0.9
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        iteration = 0
        velocity_clamp = (self.upper_bound - self.lower_bound) * 0.1

        while self.evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            self.evaluations += self.swarm_size

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            best_particle = np.argmin(scores)
            if scores[best_particle] < self.global_best_score:
                self.global_best_score = scores[best_particle]
                self.global_best_position = self.positions[best_particle]

            w = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))
            c1 = self.c1_base * (0.5 + 0.5 * np.random.rand())
            c2 = self.c2_base * (0.5 + 0.5 * np.random.rand())

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities

            # Swarm diversity control
            diversity = np.mean(np.std(self.positions, axis=0))
            if diversity < 1e-3:
                self.positions += np.random.uniform(-0.1, 0.1, self.positions.shape)

            # Dynamic mutation strategy
            if iteration % 20 == 0:
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 4, replace=False)
                mutation_strength = (0.05 + 0.45 * np.exp(-1.0 * (iteration / (self.budget // self.swarm_size))))
                mutation = np.random.normal(0, mutation_strength, (len(mutation_indices), self.dim))
                self.positions[mutation_indices] = np.clip(self.positions[mutation_indices] + mutation, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score