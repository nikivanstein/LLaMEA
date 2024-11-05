import numpy as np

class DynamicSwarmPSO:
    def __init__(self, budget, dim, initial_swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = initial_swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1, self.c2 = 2.0, 2.0
        self.w_init, self.w_final = 0.9, 0.4
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.initial_swarm_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        swarm_size = self.initial_swarm_size
        iteration = 0
        velocity_clamp = (self.upper_bound - self.lower_bound) * 0.1

        while self.evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions[:swarm_size]])
            self.evaluations += swarm_size

            better_mask = scores < self.personal_best_scores[:swarm_size]
            self.personal_best_scores[:swarm_size][better_mask] = scores[better_mask]
            self.personal_best_positions[:swarm_size][better_mask] = self.positions[:swarm_size][better_mask]

            best_particle = np.argmin(scores)
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            w = self.w_init - ((self.w_init - self.w_final) * self.evaluations / self.budget)

            r1, r2 = np.random.rand(swarm_size, self.dim), np.random.rand(swarm_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[:swarm_size] - self.positions[:swarm_size])
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[:swarm_size])
            self.velocities[:swarm_size] = w * self.velocities[:swarm_size] + cognitive_velocity + social_velocity
            self.velocities[:swarm_size] = np.clip(self.velocities[:swarm_size], -velocity_clamp, velocity_clamp)
            self.positions[:swarm_size] += self.velocities[:swarm_size]

            # Dynamic swarm size adjustment
            if iteration % 20 == 0:
                swarm_size = int(self.initial_swarm_size * (0.5 + 0.5 * np.sin(np.pi * iteration / 50)))
                if swarm_size < 5:
                    swarm_size = 5

            iteration += 1

        return self.global_best_position, self.global_best_score