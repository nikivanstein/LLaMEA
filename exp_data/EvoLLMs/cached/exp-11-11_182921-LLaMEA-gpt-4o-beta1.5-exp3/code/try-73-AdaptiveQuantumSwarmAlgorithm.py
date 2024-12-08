import numpy as np

class AdaptiveQuantumSwarmAlgorithm:
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
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.inertia_weight = 0.7
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.quantum_factor = 0.1

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

            for i in range(self.swarm_size):
                # Velocity update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])

                # Quantum-inspired particle movement
                quantum_jump = self.quantum_factor * np.random.normal(0, 1, self.dim)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                    cognitive_component + social_component + quantum_jump)

                # Position update
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Adaptive adjustment of inertia weight
            self.inertia_weight = 0.7 - 0.5 * (self.func_evaluations / self.budget)

        return self.global_best_position