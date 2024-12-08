import numpy as np

class DualSwarmAdaptivePSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5
        self.w_min, self.w_max = 0.4, 0.9
        self.k = 0.729
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
        adaptive_factor = np.ones(self.swarm_size) * 0.5
        secondary_swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        secondary_velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))

        while self.evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            secondary_scores = np.array([func(pos) for pos in secondary_swarm])
            self.evaluations += self.swarm_size * 2

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            best_particle = np.argmin(scores)
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            w = self.w_max - ((self.w_max - self.w_min) * iteration / (self.budget // (self.swarm_size * 2)))
            c1 = self.c1_max - ((self.c1_max - self.c1_min) * iteration / (self.budget // (self.swarm_size * 2)))
            c2 = self.c2_min + ((self.c2_max - self.c2_min) * iteration / (self.budget // (self.swarm_size * 2)))

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities
            
            sec_cognitive_velocity = c1 * r1 * (self.personal_best_positions - secondary_swarm)
            sec_social_velocity = c2 * r2 * (self.global_best_position - secondary_swarm)
            secondary_velocities = w * secondary_velocities + sec_cognitive_velocity + sec_social_velocity
            secondary_velocities = np.clip(secondary_velocities, -velocity_clamp, velocity_clamp)
            secondary_swarm += secondary_velocities

            if iteration % 20 == 0: 
                mutation_strength = 0.05 + 0.3 * (1 - (iteration / (self.budget // (self.swarm_size * 2))))
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 2, replace=False)
                gaussian_mutation = np.random.normal(0, mutation_strength, (len(mutation_indices), self.dim))
                self.positions[mutation_indices] = np.clip(self.positions[mutation_indices] + gaussian_mutation, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score