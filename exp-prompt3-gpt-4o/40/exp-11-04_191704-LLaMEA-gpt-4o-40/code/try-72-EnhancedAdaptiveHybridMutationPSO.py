import numpy as np

class EnhancedAdaptiveHybridMutationPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.num_swarms = 3
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1_min, self.c1_max = 1.2, 2.8  # Adjusted for exploration/exploitation balance
        self.c2_min, self.c2_max = 1.2, 2.8
        self.w_min, self.w_max = 0.3, 1.0  # Wider range for inertia weight
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

            w = self.w_max - ((self.w_max - self.w_min) * (iteration % (self.budget // (2 * self.swarm_size)) / (self.budget // (2 * self.swarm_size))))
            c1 = self.c1_max - ((self.c1_max - self.c1_min) * iteration / (self.budget // self.swarm_size))
            c2 = self.c2_min + ((self.c2_max - self.c2_min) * iteration / (self.budget // self.swarm_size))

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities

            if iteration % 10 == 0:
                self.num_swarms = np.random.randint(2, 6)
                adjacency_matrix = np.random.rand(self.swarm_size, self.swarm_size) < (1.0 / self.num_swarms)
                for i in range(self.swarm_size):
                    neighbors = adjacency_matrix[i, :]
                    neighbor_best = np.min(scores[neighbors])
                    if neighbor_best < scores[i]:
                        adaptive_factor[i] = 0.8
                    else:
                        adaptive_factor[i] = 0.2

            if iteration % 15 == 0: 
                mutation_strength = 0.05 + 0.35 * (1 - (iteration / (self.budget // self.swarm_size)))
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 3, replace=False)
                uniform_mutation = np.random.uniform(-0.3, 0.3, (len(mutation_indices), self.dim))
                gaussian_mutation = np.random.normal(0, mutation_strength, (len(mutation_indices), self.dim))
                hybrid_mutation = adaptive_factor[mutation_indices][:, None] * uniform_mutation + (1 - adaptive_factor[mutation_indices][:, None]) * gaussian_mutation
                self.positions[mutation_indices] = np.clip(self.positions[mutation_indices] + hybrid_mutation, self.lower_bound, self.upper_bound)

            if iteration % 20 == 0:
                best_positions = self.personal_best_positions[np.argsort(self.personal_best_scores)[:3]]
                for i in range(self.swarm_size):
                    if np.random.rand() < 0.1:
                        self.positions[i] = best_positions[np.random.choice(3)]

            iteration += 1

        return self.global_best_position, self.global_best_score