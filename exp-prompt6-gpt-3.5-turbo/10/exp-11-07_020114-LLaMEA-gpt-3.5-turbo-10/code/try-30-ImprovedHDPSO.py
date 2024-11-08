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
        
        # Initialization
        particles = np.random.uniform(low=-5.0, high=5.0, size=(n_particles, self.dim))
        velocities = np.zeros((n_particles, self.dim))
        best_particles = particles.copy()
        best_global = particles[np.argmin([func(p) for p in particles])]
        
        for _ in range(max_iter):
            rand_vals = np.random.rand(n_particles, 2, self.dim)
            velocities = 0.5 * velocities + c1 * rand_vals[:, 0] * (best_particles - particles) + c2 * rand_vals[:, 1] * (best_global - particles)
            particles += velocities
            particles = np.clip(particles, -5.0, 5.0)
            
            better_particles = func(particles) < func(best_particles)
            best_particles[better_particles] = particles[better_particles]
            best_global = particles[func(particles) < func(best_global)]
        
        return best_global