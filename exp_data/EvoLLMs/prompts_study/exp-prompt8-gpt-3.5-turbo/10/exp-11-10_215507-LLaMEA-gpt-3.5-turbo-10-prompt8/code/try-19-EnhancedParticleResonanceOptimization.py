class EnhancedParticleResonanceOptimization:
    def __init__(self, budget, dim, num_particles=30, w=0.5, c1=1.5, c2=1.5, elitism_rate=0.1, momentum=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.beta = 0.5  # Initialize beta
        self.elitism_rate = elitism_rate
        self.momentum = momentum

    def __call__(self, func):
        def init_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_velocity(p, g_best, particle_best, prev_velocity):
            cognitive = self.c1 * np.random.rand() * (particle_best - p)
            social = self.c2 * np.random.rand() * (g_best - p)
            velocity = self.w * prev_velocity + cognitive + social
            return velocity

        particles = init_particles()
        particle_best = particles.copy()
        g_best = particle_best[np.argmin([func(p) for p in particles])]
        velocities = np.zeros_like(particles)
        prev_velocities = np.zeros_like(particles)

        for _ in range(self.budget):
            for i in range(self.num_particles):
                self.w = np.clip(self.w * (1 - self.beta), 0.4, 0.9)
                self.beta = max(0.1, self.beta - 0.01)  # Adaptive beta update
                c1 = self.c1 * (1 - self.beta)
                c2 = self.c2 * (1 + self.beta)  # Adjust c2 based on beta
                velocities[i] = update_velocity(particles[i], g_best, particle_best[i], prev_velocities[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                if func(particles[i]) < func(particle_best[i]):
                    particle_best[i] = particles[i]
                if func(particles[i]) < func(g_best):
                    g_best = particles[i]
                prev_velocities[i] = self.momentum * velocities[i]
            elite_idx = np.argsort([func(p) for p in particles])[:int(self.elitism_rate*self.num_particles)]
            particles[elite_idx] = particle_best[elite_idx]

        return g_best