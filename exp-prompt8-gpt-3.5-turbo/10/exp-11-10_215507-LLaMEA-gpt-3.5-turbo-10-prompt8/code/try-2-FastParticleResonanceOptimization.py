import numpy as np

class FastParticleResonanceOptimization:
    def __init__(self, budget, dim, num_particles=30, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def init_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_velocity(p, g_best, particle_best, inertia_weight):
            return inertia_weight * p + self.c1 * np.random.rand() * (particle_best - p) + self.c2 * np.random.rand() * (g_best - p)

        particles = init_particles()
        particle_best = particles.copy()
        g_best = particle_best[np.argmin([func(p) for p in particles])]
        velocities = np.zeros_like(particles)
        inertia_weight = self.w_max

        for t in range(1, self.budget + 1):
            for i in range(self.num_particles):
                velocities[i] = update_velocity(particles[i], g_best, particle_best[i], inertia_weight)
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                if func(particles[i]) < func(particle_best[i]):
                    particle_best[i] = particles[i]
                if func(particles[i]) < func(g_best):
                    g_best = particles[i]
            inertia_weight = self.w_max - (self.w_max - self.w_min) * t / self.budget

        return g_best