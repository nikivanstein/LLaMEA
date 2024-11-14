import numpy as np

class Parallel_PSO_SA_Hybrid:
    def __init__(self, budget, dim, num_particles=30, max_iter=1000):
        self.budget, self.dim, self.num_particles, self.max_iter = budget, dim, num_particles, max_iter

    def __call__(self, func):
        def pso_sa_optimization():
            particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
            velocities = np.zeros((self.num_particles, self.dim))
            best_positions = particles.copy()
            best_fitness = np.full(self.num_particles, np.inf)
            global_best_position = np.zeros(self.dim)
            global_best_fitness = np.inf
            temperature = 100.0
            alpha, final_temperature = 0.99, 0.1

            for _ in range(self.max_iter):
                fitness = np.array([func(p) for p in particles])
                update_mask = fitness < best_fitness

                best_fitness[update_mask] = fitness[update_mask]
                best_positions[update_mask] = particles[update_mask]

                global_best_index = np.argmin(fitness)
                if fitness[global_best_index] < global_best_fitness:
                    global_best_fitness, global_best_position = fitness[global_best_index], particles[global_best_index]

                r_values = np.random.rand(2, self.num_particles)
                velocities = 0.5 * velocities + 2.0 * r_values[0] * (best_positions - particles) + 2.0 * r_values[1] * (global_best_position - particles)
                particles = np.clip(particles + velocities, -5.0, 5.0)

                new_positions = particles + np.random.normal(0, 1, (self.num_particles, self.dim))
                new_fitness = np.array([func(p) for p in new_positions])

                update_mask = new_fitness < fitness
                update_probabilities = np.random.rand(self.num_particles)
                accept_mask = update_mask | (update_probabilities < np.exp((fitness - new_fitness) / temperature))

                particles[accept_mask] = new_positions[accept_mask]

                temperature = np.maximum(alpha * temperature, final_temperature)

        pso_sa_optimization()
        return global_best_position