import numpy as np

class VectorizedHDPSO:
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
            r1, r2 = np.random.rand(n_particles, self.dim), np.random.rand(n_particles, self.dim)
            velocities = 0.5 * velocities + c1 * r1 * (best_particles - particles) + c2 * r2 * (best_global - particles)
            particles = np.clip(particles + velocities, -5.0, 5.0)
            
            particle_fitness = np.array([func(p) for p in particles])
            best_particle_fitness = np.array([func(bp) for bp in best_particles])
            
            update_indices = np.where(particle_fitness < best_particle_fitness)[0]
            best_particles[update_indices] = particles[update_indices]
            particle_update_indices = np.where(particle_fitness < func(best_global))[0]
            best_global = particles[particle_update_indices]
        
        return best_global