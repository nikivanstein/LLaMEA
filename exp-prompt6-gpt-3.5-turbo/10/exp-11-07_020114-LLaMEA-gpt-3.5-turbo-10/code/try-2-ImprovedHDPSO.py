import numpy as np

class ImprovedHDPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        n_particles = 30
        max_iter = self.budget // n_particles
        c1 = 2.05
        c2 = 2.05
        w_max = 0.9
        w_min = 0.4
        r1_mat = np.random.rand(max_iter, n_particles, self.dim)
        r2_mat = np.random.rand(max_iter, n_particles, self.dim)

        particles = np.random.uniform(low=-5.0, high=5.0, size=(n_particles, self.dim))
        velocities = np.zeros((n_particles, self.dim))
        best_particles = particles.copy()
        best_global = particles[np.argmin([func(p) for p in particles])]

        for _ in range(max_iter):
            velocities = 0.5 * velocities + c1 * r1_mat[_] * (best_particles - particles) + c2 * r2_mat[_] * (best_global - particles)
            particles = np.clip(particles + velocities, -5.0, 5.0)
            improved_mask = func(particles) < func(best_particles)
            best_particles[improved_mask] = particles[improved_mask]
            best_global = particles[np.argmin(func(particles))]

        return best_global