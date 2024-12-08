import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_prob = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate current position
                score = func(self.particles[i])
                self.func_evaluations += 1

                # Update personal bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

                # Update velocities
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i]
                    + self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i])
                    + self.social_coeff * r2 * (self.global_best_position - self.particles[i])
                )

                # Quantum tunneling
                if np.random.rand() < self.quantum_prob:
                    self.velocities[i] += np.random.normal(0, 1, self.dim)

                # Update particles positions
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Dynamic adjustment of inertia weight and quantum probability
            self.inertia_weight = 0.7 - 0.4 * (self.func_evaluations / self.budget)
            self.quantum_prob = 0.1 * np.sin(2 * np.pi * self.func_evaluations / self.budget)

        return self.global_best_position