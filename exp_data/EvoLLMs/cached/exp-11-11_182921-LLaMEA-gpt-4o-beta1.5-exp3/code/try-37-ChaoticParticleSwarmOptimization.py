import numpy as np

class ChaoticParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.phi1 = 2.05
        self.phi2 = 2.05
        self.k = 10
        self.chaos_control_param = 0.7

    def chaotic_map(self, x):
        return self.chaos_control_param * x * (1 - x)

    def __call__(self, func):
        chaotic_sequence = np.random.rand(self.swarm_size)
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate fitness
                score = func(self.positions[i])
                self.func_evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.positions[i])

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.positions[i])

            for i in range(self.swarm_size):
                # Chaotic update
                chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i])
                inertia_weight = 0.5 + chaotic_sequence[i] / 2.0

                # Adaptive topology: neighborhood influence
                neighbors_indices = np.random.choice(self.swarm_size, self.k, replace=False)
                neighborhood_best_position = min(neighbors_indices, key=lambda idx: self.personal_best_scores[idx])

                # Update velocity
                cognitive_component = self.phi1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.phi2 * np.random.rand(self.dim) * (self.personal_best_positions[neighborhood_best_position] - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component

                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position