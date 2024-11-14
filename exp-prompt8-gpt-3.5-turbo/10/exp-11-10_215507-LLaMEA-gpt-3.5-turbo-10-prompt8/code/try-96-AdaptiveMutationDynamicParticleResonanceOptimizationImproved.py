import numpy as np

class AdaptiveMutationDynamicParticleResonanceOptimizationImproved:
    def __init__(self, budget, dim, num_particles=30, w=0.5, c1=1.5, c2=1.5, elitism_rate=0.1, momentum=0.1, diversity_rate=0.1, inertia_range=(0.1, 0.9)):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.elitism_rate = elitism_rate
        self.momentum = momentum
        self.diversity_rate = diversity_rate
        self.inertia_range = inertia_range

    def __call__(self, func):
        def mutate(particle, diversity_rate):
            return particle + np.random.normal(0, diversity_rate, size=self.dim)

        particles = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        particle_best = particles.copy()
        g_best = particle_best[np.argmin([func(p) for p in particles])]
        velocities = np.zeros_like(particles)
        prev_velocities = np.zeros_like(particles)

        for t in range(1, self.budget + 1):
            inertia_weight = self.inertia_range[0] + (self.inertia_range[1] - self.inertia_range[0]) * (t / self.budget)
            diversity_rate = self.diversity_rate * np.std(particles, axis=0)
            for i in range(self.num_particles):
                velocities[i] = inertia_weight * velocities[i] + self.c1 * np.random.rand() * (particle_best[i] - particles[i]) + self.c2 * np.random.rand() * (g_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                particles[i] = mutate(particles[i], diversity_rate)
                if func(particles[i]) < func(particle_best[i]):
                    particle_best[i] = particles[i]
                if func(particles[i]) < func(g_best):
                    g_best = particles[i]
                prev_velocities[i] = self.momentum * velocities[i]
            elite_idx = np.argsort([func(p) for p in particles])[:int(self.elitism_rate * self.num_particles)]
            particles[elite_idx] = particle_best[elite_idx]

        return g_best