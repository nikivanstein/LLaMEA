class AdaptiveInertiaParticleResonanceOptimization:
    def __init__(self, budget, dim, num_particles=30, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def init_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_velocity(p, g_best, particle_best, w):
            return w * p + self.c1 * np.random.rand() * (particle_best - p) + self.c2 * np.random.rand() * (g_best - p)

        particles = init_particles()
        particle_best = particles.copy()
        g_best = particle_best[np.argmin([func(p) for p in particles])]
        velocities = np.zeros_like(particles)
        w = 0.9  # Initial inertia weight

        for t in range(1, self.budget + 1):
            for i in range(self.num_particles):
                velocities[i] = update_velocity(particles[i], g_best, particle_best[i], w)
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                if func(particles[i]) < func(particle_best[i]):
                    particle_best[i] = particles[i]
                if func(particles[i]) < func(g_best):
                    g_best = particles[i]
            w = 0.4 + 0.5 * (self.budget - t) / self.budget

        return g_best