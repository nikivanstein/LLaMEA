import numpy as np

class AdaptiveParticleSwarmOptimizationQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_score = float('inf')
        self.global_best_position = None
        self.func_evaluations = 0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.p_quantum = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate fitness
                score = func(self.positions[i])
                self.func_evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[i]
            
            # Adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (self.func_evaluations / self.budget)

            for i in range(self.swarm_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component

                # Quantum-inspired perturbation
                if np.random.rand() < self.p_quantum:
                    self.velocities[i] += np.random.normal(0, 0.1, self.dim)

                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position