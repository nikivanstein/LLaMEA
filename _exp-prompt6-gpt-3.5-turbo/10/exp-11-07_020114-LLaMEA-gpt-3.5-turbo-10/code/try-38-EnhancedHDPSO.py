import numpy as np

class EnhancedHDPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        n_particles = 30
        max_iter = self.budget // n_particles
        c1 = 2.05
        c2 = 2.05
        w = 0.9
        w_decay = (0.9 - 0.4) / max_iter  # Linearly decreasing inertia weight
        
        particles = np.random.uniform(low=-5.0, high=5.0, size=(n_particles, self.dim))
        velocities = np.zeros((n_particles, self.dim))
        best_particles = particles.copy()
        best_global = particles[np.argmin([func(p) for p in particles])]

        for _ in range(max_iter):
            for i in range(n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (best_particles[i] - particles[i]) + c2 * r2 * (best_global - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)
                
                particle_fitness = func(particles[i])
                best_particle_fitness = func(best_particles[i])
                if particle_fitness < best_particle_fitness:
                    best_particles[i] = particles[i]
                    if particle_fitness < func(best_global):
                        best_global = particles[i]

            w -= w_decay  # Update inertia weight

        return best_global