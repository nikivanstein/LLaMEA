import numpy as np

class EnhancedAdaptivePSO:
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

    def levy_flight(self, dim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / abs(v)**(1 / beta)
        return step

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
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            w = 0.5
            c1, c2 = 1.5, 1.5
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities

            # Dynamic neighborhood adaptation
            if iteration % 7 == 0:
                adj_matrix = np.random.rand(self.swarm_size, self.swarm_size) < (1.0 / np.random.randint(1, 4))
                for i in range(self.swarm_size):
                    neighbors = adj_matrix[i, :]
                    if np.any(neighbors):
                        neighbor_best = np.min(scores[neighbors])
                        if neighbor_best < scores[i]:
                            self.positions[i] += self.levy_flight(self.dim)

            # Levy flight mutation
            if iteration % 20 == 0:
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 3, replace=False)
                for idx in mutation_indices:
                    self.positions[idx] = np.clip(self.positions[idx] + self.levy_flight(self.dim), self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score