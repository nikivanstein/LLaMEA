import numpy as np

class DynamicMultiSwarmPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1, self.c2 = 1.5, 2.5
        self.w = 0.7
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0
        self.num_groups = 3

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

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities

            if iteration % 10 == 0:
                self.num_groups = np.random.randint(2, 5)
                for i in range(self.swarm_size):
                    group_id = i % self.num_groups
                    local_best = np.min(scores[group_id::self.num_groups])
                    if local_best < scores[i]:
                        self.velocities[i] *= 0.7

            if iteration % 15 == 0:
                mutation_strength = 0.05 + 0.3 * (1 - (iteration / (self.budget // self.swarm_size)))
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 3, replace=False)
                mutation = np.random.normal(0, mutation_strength, (len(mutation_indices), self.dim))
                self.positions[mutation_indices] = np.clip(self.positions[mutation_indices] + mutation, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score