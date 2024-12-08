import numpy as np

class AdaptiveSelfAdaptivePSO:
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
        self.inertia_weights = np.random.uniform(0.4, 0.9, self.swarm_size)
        self.cognitive_factors = np.random.uniform(1.5, 2.5, self.swarm_size)
        self.social_factors = np.random.uniform(1.5, 2.5, self.swarm_size)

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

            # Self-adaptive parameters
            self.inertia_weights += np.random.normal(0, 0.1, self.swarm_size)
            self.cognitive_factors += np.random.normal(0, 0.1, self.swarm_size)
            self.social_factors += np.random.normal(0, 0.1, self.swarm_size)
            self.inertia_weights = np.clip(self.inertia_weights, 0.4, 0.9)
            self.cognitive_factors = np.clip(self.cognitive_factors, 1.5, 2.5)
            self.social_factors = np.clip(self.social_factors, 1.5, 2.5)

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = self.cognitive_factors[:, None] * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.social_factors[:, None] * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weights[:, None] * (self.velocities + cognitive_velocity + social_velocity)
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities

            # Adaptive topology and mutation strategy
            if iteration % 10 == 0:
                adjacency_matrix = np.random.rand(self.swarm_size, self.swarm_size) < (1.0 / np.random.randint(1, 5))
                for i in range(self.swarm_size):
                    neighbors = adjacency_matrix[i, :]
                    neighbor_best = np.min(scores[neighbors])
                    if neighbor_best < scores[i]:
                        self.inertia_weights[i] = 0.7

            if iteration % 15 == 0:
                mutation_strength = 0.02 + 0.38 * (1 - (iteration / (self.budget // self.swarm_size)))
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 3, replace=False)
                gaussian_mutation = np.random.normal(0, mutation_strength, (len(mutation_indices), self.dim))
                self.positions[mutation_indices] = np.clip(self.positions[mutation_indices] + gaussian_mutation, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score