import numpy as np

class EnhancedDynamicMultiSwarmPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.num_swarms = 3
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.5
        self.c2 = 2.5
        self.w_max, self.w_min = 0.9, 0.4
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.sub_swarm_best_positions = np.zeros((self.num_swarms, self.dim))
        self.sub_swarm_best_scores = np.full(self.num_swarms, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        iteration = 0
        swarm_indices = np.array_split(np.arange(self.swarm_size), self.num_swarms)
        while self.evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            self.evaluations += self.swarm_size

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            for i, idx in enumerate(swarm_indices):
                sub_scores = scores[idx]
                sub_best_particle = np.argmin(sub_scores)
                if sub_scores[sub_best_particle] < self.sub_swarm_best_scores[i]:
                    self.sub_swarm_best_scores[i] = sub_scores[sub_best_particle]
                    self.sub_swarm_best_positions[i] = self.positions[idx[sub_best_particle]]

            if np.min(self.sub_swarm_best_scores) < self.global_best_score:
                best_swarm_index = np.argmin(self.sub_swarm_best_scores)
                self.global_best_score = self.sub_swarm_best_scores[best_swarm_index]
                self.global_best_position = self.sub_swarm_best_positions[best_swarm_index]

            w = self.w_max - ((self.w_max - self.w_min) * (iteration / (self.budget // self.swarm_size)))
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.positions += self.velocities

            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Dynamic interaction among sub-swarms
            if iteration % 5 == 0:
                for i, indices in enumerate(swarm_indices):
                    if np.random.rand() < 0.3:
                        self.positions[indices] += np.random.normal(0, 0.1, self.positions[indices].shape)
                        self.positions[indices] = np.clip(self.positions[indices], self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score