import numpy as np

class Enhanced_PSO_SA_Hybrid:
    def __init__(self, budget, dim, num_particles=30, max_iter=1000):
        self.budget, self.dim, self.num_particles, self.max_iter = budget, dim, num_particles, max_iter

    def __call__(self, func):
        particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        best_positions = particles.copy()
        best_fitness = np.full(self.num_particles, np.inf)
        global_best_position = np.zeros(self.dim)
        global_best_fitness = np.inf
        temperature = 100.0
        alpha, final_temperature = 0.99, 0.1

        for _ in range(self.max_iter):
            fitness = func(particles)
            update_particles = fitness < best_fitness

            best_positions[update_particles] = particles[update_particles]
            best_fitness[update_particles] = fitness[update_particles]

            update_global_best = fitness < global_best_fitness
            global_best_position = np.where(update_global_best, particles, global_best_position)
            global_best_fitness = np.where(update_global_best, fitness, global_best_fitness)

            r1, r2 = np.random.rand(2, self.num_particles, 1)
            velocities = 0.5 * velocities + 2.0 * r1 * (best_positions - particles) + 2.0 * r2 * (global_best_position - particles)
            particles = np.clip(particles + velocities, -5.0, 5.0)

            new_positions = particles + np.random.normal(0, 1, (self.num_particles, self.dim))
            new_fitness = func(new_positions)

            update_particles = new_fitness < fitness
            fitness = np.where(update_particles, new_fitness, fitness)
            particles = np.where(update_particles[..., np.newaxis], new_positions, particles)

            prob_accept = np.random.rand(self.num_particles) < np.exp((fitness - new_fitness) / temperature)
            particles = np.where(prob_accept[..., np.newaxis], new_positions, particles)

            temperature = np.maximum(alpha * temperature, final_temperature)

        return global_best_position