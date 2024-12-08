class EnhancedHybridDynamicFireflyOptimizationImproved:
    def __init__(self, budget, dim, num_particles=30, w=0.5, c1=1.5, c2=1.5, elitism_rate=0.1, momentum=0.1, mutation_rate=0.1, inertia_range=(0.1, 0.9), alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.elitism_rate = elitism_rate
        self.momentum = momentum
        self.mutation_rate = mutation_rate
        self.inertia_range = inertia_range
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        def firefly_move(particle, best_particle):
            beta = self.beta0 * np.exp(-self.gamma * np.linalg.norm(particle - best_particle))
            return particle + beta * (np.random.rand(self.dim) - 0.5)

        particles = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        particle_best = particles.copy()
        g_best = particle_best[np.argmin([func(p) for p in particles])]
        velocities = np.zeros_like(particles)
        prev_velocities = np.zeros_like(particles)

        for t in range(1, self.budget + 1):
            inertia_weight = self.inertia_range[0] + (self.inertia_range[1] - self.inertia_range[0]) * (t / self.budget)
            for i in range(self.num_particles):
                velocities[i] = inertia_weight * velocities[i] + self.c1 * np.random.rand() * (particle_best[i] - particles[i]) + self.c2 * np.random.rand() * (g_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                particles[i] = firefly_move(particles[i], g_best)
                particles[i] = mutate(particles[i])
                if func(particles[i]) < func(particle_best[i]):
                    particle_best[i] = particles[i]
                if func(particles[i]) < func(g_best):
                    g_best = particles[i]
                prev_velocities[i] = self.momentum * velocities[i]
            elite_idx = np.argsort([func(p) for p in particles])[:int(self.elitism_rate * self.num_particles)]
            particles[elite_idx] = particle_best[elite_idx]

        return g_best