import numpy as np

class EnhancedDynamicSwarmPSO:
    def __init__(self, budget, dim, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1_min, self.c1_max = 1.3, 2.7
        self.c2_min, self.c2_max = 1.3, 2.7
        self.w_min, self.w_max = 0.3, 0.8
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
            best_particle_score = scores[best_particle]
            if best_particle_score < self.global_best_score:
                self.global_best_score = best_particle_score
                self.global_best_position = self.positions[best_particle]

            w = self.w_max - ((self.w_max - self.w_min) * iteration / (self.budget // self.swarm_size))
            c1 = self.c1_max - ((self.c1_max - self.c1_min) * iteration / (self.budget // self.swarm_size))
            c2 = self.c2_min + ((self.c2_max - self.c2_min) * iteration / (self.budget // self.swarm_size))

            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            cognitive_velocity = c1 * r1 * (self.personal_best_positions - self.positions)
            social_velocity = c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = w * self.velocities + cognitive_velocity + social_velocity
            self.velocities = np.clip(self.velocities, -velocity_clamp, velocity_clamp)
            self.positions += self.velocities
            
            # Dynamic swarm resizing
            if iteration % 5 == 0:
                self.swarm_size = np.random.randint(20, 40)
                new_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
                self.positions = np.vstack((self.positions, new_positions))[:self.swarm_size]
                self.velocities = np.clip(self.velocities, -1, 1)[:self.swarm_size]
                self.personal_best_positions = self.personal_best_positions[:self.swarm_size]
                self.personal_best_scores = np.append(self.personal_best_scores, np.full(new_positions.shape[0], float('inf')))[:self.swarm_size]

            # Hybrid mutation strategy
            if iteration % 12 == 0: 
                mutation_strength = 0.01 + 0.39 * (1 - (iteration / (self.budget // self.swarm_size)))
                mutation_indices = np.random.choice(self.swarm_size, self.swarm_size // 4, replace=False)
                uniform_mutation = np.random.uniform(-0.6, 0.6, (len(mutation_indices), self.dim))
                gaussian_mutation = np.random.normal(0, mutation_strength, (len(mutation_indices), self.dim))
                hybrid_mutation = 0.5 * uniform_mutation + 0.5 * gaussian_mutation
                self.positions[mutation_indices] = np.clip(self.positions[mutation_indices] + hybrid_mutation, self.lower_bound, self.upper_bound)

            iteration += 1

        return self.global_best_position, self.global_best_score