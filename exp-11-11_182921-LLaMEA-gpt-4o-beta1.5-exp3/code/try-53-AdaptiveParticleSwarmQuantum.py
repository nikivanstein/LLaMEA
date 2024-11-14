import numpy as np

class AdaptiveParticleSwarmQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 15 + int(2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.zeros((self.swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w_max = 0.9
        self.w_min = 0.4
        self.tau = 0.2  # Quantum perturbation probability

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                current_score = func(self.positions[i])
                self.func_evaluations += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]

            inertia_weight = self.w_max - (self.w_max - self.w_min) * (self.func_evaluations / self.budget)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity

                # Quantum-inspired perturbation
                if np.random.rand() < self.tau:
                    self.velocities[i] += np.random.normal(0, 0.1, self.dim)

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position