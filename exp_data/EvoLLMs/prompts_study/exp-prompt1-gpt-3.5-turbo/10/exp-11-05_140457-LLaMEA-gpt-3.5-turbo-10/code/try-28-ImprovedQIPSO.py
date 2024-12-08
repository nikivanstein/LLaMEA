import numpy as np

class ImprovedQIPSO(QIPSO):
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, quantum_factor=0.5, quantum_factor_decay=0.95):
        super().__init__(budget, dim, num_particles, inertia_weight, cognitive_weight, social_weight, quantum_factor)
        self.quantum_factor_decay = quantum_factor_decay
        self.chaos_prob = 0.05

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

                if np.random.rand() < self.chaos_prob:  # Introduce chaos
                    chaos_factor = np.random.uniform(-1, 1, size=self.dim)
                    velocities[i] = velocities[i] + chaos_factor

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

        return global_best