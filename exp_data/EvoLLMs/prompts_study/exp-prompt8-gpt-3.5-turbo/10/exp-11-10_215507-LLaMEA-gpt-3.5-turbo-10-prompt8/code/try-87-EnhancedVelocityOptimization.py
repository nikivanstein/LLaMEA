import numpy as np

class EnhancedVelocityOptimization(AdaptiveMutationDynamicParticleResonanceOptimizationImproved):
    def __call__(self, func):
        def mutate(particle, diversity_rate):
            return particle + np.random.normal(0, diversity_rate, size=self.dim)

        particles = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        particle_best = particles.copy()
        g_best = particle_best[np.argmin([func(p) for p in particles])]
        velocities = np.zeros_like(particles)
        prev_velocities = np.zeros_like(particles)
        memory = np.zeros_like(particles)

        for t in range(1, self.budget + 1):
            inertia_weight = self.inertia_range[0] + (self.inertia_range[1] - self.inertia_range[0]) * (t / self.budget)
            for i in range(self.num_particles):
                velocities[i] = inertia_weight * velocities[i] + self.c1 * np.random.rand() * (particle_best[i] - particles[i]) + self.c2 * np.random.rand() * (g_best - particles[i]) + 0.1 * memory[i]
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                particles[i] = mutate(particles[i], self.diversity_rate * np.std(particles, axis=0))
                if func(particles[i]) < func(particle_best[i]):
                    particle_best[i] = particles[i]
                if func(particles[i]) < func(g_best):
                    g_best = particles[i]
                prev_velocities[i] = self.momentum * velocities[i]
                memory[i] = np.where(func(particles[i]) < func(particle_best[i]), 0.1 * memory[i] + 0.9 * (particle_best[i] - particles[i]), memory[i])
            elite_idx = np.argsort([func(p) for p in particles])[:int(self.elitism_rate * self.num_particles)]
            particles[elite_idx] = particle_best[elite_idx]

        return g_best