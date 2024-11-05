import numpy as np

class EnhancedDualPhasePSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.num_swarms = 3
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5
        self.w_min, self.w_max = 0.2, 0.9
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
        phase_switch = self.budget // (2 * self.swarm_size)
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

            if iteration < phase_switch:
                w = self.w_min + ((self.w_max - self.w_min) * iteration / phase_switch)
                c1, c2 = self.c1_min, self.c2_max
            else:
                w = self.w_max - ((self.w_max - self.w_min) * (iteration - phase_switch) / phase_switch)
                c1, c2 = self.c2_min, self.c1_max

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = self.k * (w * self.velocities + cognitive_velocity + social_velocity)

            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            if iteration % 15 == 0:
                mutation_strength = 0.1 + 0.4 * np.random.rand()
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 4, replace=False)
                directions = np.random.choice([-1, 1], (len(mutation_indices), self.dim))
                self.positions[mutation_indices] += directions * mutation_strength
                self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score