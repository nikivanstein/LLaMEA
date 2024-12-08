import numpy as np

class AdaptiveQuantumInspiredPSO:
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
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.inertia_weight = 0.7
        self.quantum_potential = 0.2

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate fitness
                current_score = func(self.positions[i])
                self.func_evaluations += 1

                # Update personal best
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]

                # Adaptive inertia weight update
                self.inertia_weight = 0.7 - 0.3 * (self.func_evaluations / self.budget)

            # Update velocities and positions
            for i in range(self.swarm_size):
                cognitive_component = self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])

                # Apply quantum-inspired perturbation
                quantum_perturbation = self.quantum_potential * np.random.normal(0, 1, self.dim)
                
                # Update velocity
                self.velocities[i] = (self.inertia_weight * self.velocities[i] + cognitive_component + social_component + quantum_perturbation)
                self.positions[i] += self.velocities[i]
                
                # Ensure particles remain within bounds
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position