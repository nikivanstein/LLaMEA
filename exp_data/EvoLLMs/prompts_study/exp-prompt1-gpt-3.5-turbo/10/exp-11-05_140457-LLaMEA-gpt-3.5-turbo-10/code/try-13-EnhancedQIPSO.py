import numpy as np

class EnhancedQIPSO(QIPSO):
    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        particles = initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        best_positions = particles.copy()
        best_values = np.array([func(p) for p in best_positions])
        global_best_idx = np.argmin(best_values)
        global_best = best_positions[global_best_idx].copy()

        for _ in range(self.budget):
            for i in range(self.num_particles):
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (best_positions[i] - particles[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (global_best - particles[i])
                quantum_influence = self.quantum_factor * np.random.uniform(-1, 1, size=self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_component + social_component + quantum_influence
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                current_value = func(particles[i])
                if current_value < best_values[i]:
                    best_values[i] = current_value
                    best_positions[i] = particles[i].copy()
                    if current_value < best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = particles[i].copy()
                        if np.random.rand() < 0.1:  # 10% chance
                            self.quantum_factor *= self.quantum_factor_decay

            # Incorporating differential evolution to update a subset of particles
            subset_idx = np.random.choice(self.num_particles, int(0.1 * self.num_particles), replace=False)
            for idx in subset_idx:
                r1, r2, r3 = np.random.choice(self.num_particles, 3, replace=False)
                mutant = particles[r1] + 0.5*(particles[r2] - particles[r3])
                crossover = np.random.rand(self.dim) < 0.9
                particles[idx] = np.where(crossover, mutant, particles[idx])

        return global_best