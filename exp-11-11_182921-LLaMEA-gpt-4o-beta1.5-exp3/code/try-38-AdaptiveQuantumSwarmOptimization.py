import numpy as np

class AdaptiveQuantumSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(15 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, dim))
        self.func_evaluations = 0
        self.best_particle_score = np.full(self.swarm_size, float('inf'))
        self.best_particle_position = np.copy(self.positions)
        self.global_best_score = float('inf')
        self.global_best_position = None
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.tau = 0.1  # Quantum influence on velocity

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                current_score = func(self.positions[i])
                self.func_evaluations += 1

                if current_score < self.best_particle_score[i]:
                    self.best_particle_score[i] = current_score
                    self.best_particle_position[i] = self.positions[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_term = self.cognitive_coefficient * r1 * (self.best_particle_position[i] - self.positions[i])
                social_term = self.social_coefficient * r2 * (self.global_best_position - self.positions[i])
                
                # Quantum-inspired velocity perturbation
                quantum_term = np.random.normal(0, 1, self.dim) if np.random.rand() < self.tau else 0
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_term +
                                      social_term +
                                      quantum_term)

                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Adaptive adjustment of inertia weight and tau
            self.inertia_weight = 0.7 - 0.4 * (self.func_evaluations / self.budget)
            self.tau = 0.1 * (1 - np.cos(2 * np.pi * self.func_evaluations / self.budget))

        return self.global_best_position