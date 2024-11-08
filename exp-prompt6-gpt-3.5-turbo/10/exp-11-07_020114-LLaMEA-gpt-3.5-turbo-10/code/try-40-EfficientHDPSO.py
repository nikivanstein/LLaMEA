import numpy as np

class EfficientHDPSO:
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
        
        particles = np.random.uniform(low=-5.0, high=5.0, size=(n_particles, self.dim))
        velocities = np.zeros((n_particles, self.dim))
        best_particles = np.copy(particles)
        best_global = particles[np.argmin([func(p) for p in particles])]
        
        for _ in range(max_iter):
            r1, r2 = np.random.rand(self.dim, 2, n_particles)
            velocities = 0.5 * velocities + c1 * r1.T * (best_particles - particles) + c2 * r2.T * (best_global - particles)
            particles = np.clip(particles + velocities, -5.0, 5.0)
            
            particle_fitness = func(particles.T)
            best_particle_fitness = func(best_particles.T)
            update_indices = np.where(particle_fitness < best_particle_fitness)[0]
            
            best_particles[update_indices] = particles[update_indices]
            better_global_idx = np.argmin(particle_fitness)
            if particle_fitness[better_global_idx] < func(best_global):
                best_global = particles[better_global_idx]
        
        return best_global