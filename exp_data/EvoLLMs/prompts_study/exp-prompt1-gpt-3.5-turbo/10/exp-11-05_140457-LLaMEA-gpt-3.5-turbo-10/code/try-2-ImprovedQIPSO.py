import numpy as np

class ImprovedQIPSO(QIPSO):
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5, quantum_factor=0.5):
        super().__init__(budget, dim, num_particles, inertia_weight, cognitive_weight, social_weight, quantum_factor)
        self.min_inertia = 0.1
        self.max_inertia = 0.9

    def __call__(self, func):
        def adapt_inertia(iteration):
            return self.min_inertia + (self.max_inertia - self.min_inertia) * (self.budget - iteration) / self.budget

        particles = initialize_particles()
        velocities = np.zeros((self.num_particles, self.dim))
        best_positions = particles.copy()
        best_values = np.array([func(p) for p in best_positions])
        global_best_idx = np.argmin(best_values)
        global_best = best_positions[global_best_idx].copy()

        for iter_count in range(self.budget):
            inertia = adapt_inertia(iter_count)
            for i in range(self.num_particles):
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (best_positions[i] - particles[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (global_best - particles[i])
                quantum_influence = self.quantum_factor * np.random.uniform(-1, 1, size=self.dim)
                velocities[i] = inertia * velocities[i] + cognitive_component + social_component + quantum_influence
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                current_value = func(particles[i])
                if current_value < best_values[i]:
                    best_values[i] = current_value
                    best_positions[i] = particles[i].copy()
                    if current_value < best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = particles[i].copy()

        return global_best