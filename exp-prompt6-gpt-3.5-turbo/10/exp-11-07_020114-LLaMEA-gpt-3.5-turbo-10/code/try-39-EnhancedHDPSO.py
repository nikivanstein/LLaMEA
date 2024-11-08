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
        
        # Update particle velocity and position
        def update_particle(particle, velocity, best_particle, best_global):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = 0.5 * velocity + c1 * r1 * (best_particle - particle) + c2 * r2 * (best_global - particle)
            particle = np.clip(particle + velocity, -5.0, 5.0)
            return particle, velocity
        
        # Initialization
        particles, velocities, best_particles = initialize_particles()
        best_global = particles[np.argmin([func(p) for p in particles])]
        
        for _ in range(max_iter):
            particles, velocities = zip(*[update_particle(p, v, bp, best_global) for p, v, bp in zip(particles, velocities, best_particles)])
            particle_fitness = np.array([func(p) for p in particles])
            best_particle_fitness = np.array([func(bp) for bp in best_particles])
            
            update_indices = np.where(particle_fitness < best_particle_fitness)[0]
            best_particles[update_indices] = particles[update_indices]
            best_global = particles[np.argmin(particle_fitness)]

        return best_global