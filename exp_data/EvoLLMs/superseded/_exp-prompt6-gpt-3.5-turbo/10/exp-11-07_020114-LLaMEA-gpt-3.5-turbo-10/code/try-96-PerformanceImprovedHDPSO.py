import numpy as np

class PerformanceImprovedHDPSO:
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
        best_particles = particles.copy()
        best_global = particles[np.argmin([func(p) for p in particles])]
        
        for _ in range(max_iter):
            r1, r2 = np.random.rand(n_particles, self.dim), np.random.rand(n_particles, self.dim)
            velocities = 0.5 * velocities + c1 * r1 * (best_particles - particles) + c2 * r2 * (best_global - particles)
            np.clip(particles + velocities, -5.0, 5.0, out=particles)  # Optimizing particle position update
            
            particle_fitness = np.array([func(p) for p in particles])
            update_particles = np.where(particle_fitness < np.array([func(bp) for bp in best_particles]))[0]
            best_particles[update_particles] = particles[update_particles]
            
            update_global = np.argmin(particle_fitness)
            if particle_fitness[update_global] < func(best_global):
                best_global = particles[update_global]
        
        return best_global