import numpy as np

class EnhancedMultiSwarmPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.num_swarms = 4  # Increase diversity with more sub-swarms
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1, self.c2 = 2.0, 2.0  # Fix cognitive and social coefficients for simplicity
        self.k = 0.729
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
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
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            w = np.exp(-0.5 * (iteration / (self.budget // self.swarm_size)))  # Adaptive inertia weight
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.k * (w * self.velocities + cognitive_velocity + social_velocity)
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities

            # Multi-swarm strategy
            if iteration % 5 == 0:
                swarms = np.array_split(np.arange(self.swarm_size), self.num_swarms)
                for swarm in swarms:
                    swarm_scores = scores[swarm]
                    swarm_best_idx = swarm[np.argmin(swarm_scores)]
                    if scores[swarm_best_idx] < self.global_best_score:
                        self.global_best_score = scores[swarm_best_idx]
                        self.global_best_position = self.positions[swarm_best_idx]

            # Lévy flight mutation
            if iteration % 20 == 0:
                mutation_strength = np.random.standard_cauchy((self.swarm_size, self.dim)) * 0.1
                self.positions += mutation_strength
                self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score