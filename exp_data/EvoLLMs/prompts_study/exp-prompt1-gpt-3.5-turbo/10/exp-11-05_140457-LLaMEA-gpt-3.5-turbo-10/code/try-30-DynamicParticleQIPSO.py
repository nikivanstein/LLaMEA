import numpy as np

class DynamicParticleQIPSO(ImprovedQIPSO):
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, quantum_factor=0.5, quantum_factor_decay=0.95, min_particles=10, max_particles=50, decay_threshold=0.2, growth_threshold=0.8):
        super().__init__(budget, dim, num_particles, inertia_weight, cognitive_weight, social_weight, quantum_factor, quantum_factor_decay)
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.decay_threshold = decay_threshold
        self.growth_threshold = growth_threshold

    def __call__(self, func):
        def initialize_particles(num_particles):
            return np.random.uniform(-5.0, 5.0, size=(num_particles, self.dim))

        num_particles = self.num_particles
        particles = initialize_particles(num_particles)
        velocities = np.zeros((num_particles, self.dim))
        best_positions = particles.copy()
        best_values = np.array([func(p) for p in best_positions])
        global_best_idx = np.argmin(best_values)
        global_best = best_positions[global_best_idx].copy()

        for _ in range(self.budget):
            for i in range(num_particles):
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

            # Dynamically adjust number of particles
            if np.random.rand() < self.decay_threshold and num_particles > self.min_particles:
                num_particles -= 1
                particles = np.delete(particles, -1, axis=0)
                velocities = np.delete(velocities, -1, axis=0)
                best_positions = np.delete(best_positions, -1, axis=0)
                best_values = np.delete(best_values, -1)
            elif np.random.rand() < self.growth_threshold and num_particles < self.max_particles:
                num_particles += 1
                particles = np.vstack((particles, np.random.uniform(-5.0, 5.0, size=(1, self.dim)))
                velocities = np.vstack((velocities, np.zeros(self.dim)))
                best_positions = np.vstack((best_positions, particles[-1].copy()))
                best_values = np.append(best_values, func(particles[-1]))

        return global_best