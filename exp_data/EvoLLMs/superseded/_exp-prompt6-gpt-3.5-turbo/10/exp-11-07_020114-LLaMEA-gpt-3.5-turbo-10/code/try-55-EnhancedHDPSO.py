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
        w_max = 0.9
        w_min = 0.4
        
        def initialize_particles():
            return np.random.uniform(low=-5.0, high=5.0, size=(n_particles, self.dim)), np.zeros((n_particles, self.dim)), np.zeros((n_particles, self.dim))
        
        particles, velocities, best_particles = initialize_particles()
        best_global = particles[np.argmin([func(p) for p in particles])
        
        for _ in range(max_iter):
            r1, r2 = np.random.rand(n_particles, self.dim, 2)
            velocities = 0.5 * velocities + c1 * r1 * (best_particles - particles)[:, :, np.newaxis] + c2 * r2 * (best_global - particles)[:, :, np.newaxis]
            particles = np.clip(particles + velocities, -5.0, 5.0)
            
            fitness_values = np.array([func(p) for p in particles])
            improved_particles_mask = fitness_values < np.array([func(bp) for bp in best_particles])
            
            best_particles[improved_particles_mask] = particles[improved_particles_mask]
            best_global = particles[np.argmin(fitness_values)]
        
        return best_global