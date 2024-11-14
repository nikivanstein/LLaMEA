import numpy as np

class AdaptiveParticleSwarmQuantumTunneling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.zeros((self.swarm_size, dim))
        self.best_local_positions = np.copy(self.positions)
        self.best_local_scores = np.full(self.swarm_size, float('inf'))
        self.best_global_position = None
        self.best_global_score = float('inf')
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.inertia_weight = 0.9
        self.func_evaluations = 0
        self.tunneling_rate = 0.05  # Probability of quantum tunneling

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate fitness
                score = func(self.positions[i])
                self.func_evaluations += 1
                
                # Update personal best
                if score < self.best_local_scores[i]:
                    self.best_local_scores[i] = score
                    self.best_local_positions[i] = self.positions[i]

                # Update global best
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            for i in range(self.swarm_size):
                cognitive_component = self.c1 * r1 * (self.best_local_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.best_global_position - self.positions[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component

                # Quantum tunneling
                if np.random.rand() < self.tunneling_rate:
                    self.velocities[i] += np.random.normal(0, 1, self.dim)

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Adaptive inertia weight
            self.inertia_weight = 0.4 + 0.5 * (1 - self.func_evaluations / self.budget)

        return self.best_global_position