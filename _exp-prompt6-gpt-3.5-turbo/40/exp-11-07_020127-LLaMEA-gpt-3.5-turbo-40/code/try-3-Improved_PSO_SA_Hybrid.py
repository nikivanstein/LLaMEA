import numpy as np

class Improved_PSO_SA_Hybrid:
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
                fitness_values = np.apply_along_axis(func, 1, particles)

                update_indices = fitness_values < best_fitness
                best_fitness[update_indices] = fitness_values[update_indices]
                best_positions[update_indices] = particles[update_indices]

                global_best_index = np.argmin(fitness_values)
                if fitness_values[global_best_index] < global_best_fitness:
                    global_best_fitness, global_best_position = fitness_values[global_best_index], particles[global_best_index]

                r_values = np.random.rand(2, self.num_particles)
                velocities = 0.5 * velocities + 2.0 * r_values[0] * (best_positions - particles) + 2.0 * r_values[1] * (global_best_position - particles)
                particles = np.clip(particles + velocities, -5.0, 5.0)

                new_positions = particles + np.random.normal(0, 1, (self.num_particles, self.dim))
                new_fitness_values = np.apply_along_axis(func, 1, new_positions)

                accept_probabilities = np.exp((fitness_values - new_fitness_values) / temperature)
                acceptance_mask = new_fitness_values < fitness_values + np.random.rand(self.num_particles) * accept_probabilities

                particles = np.where(acceptance_mask[:, None], new_positions, particles)

                temperature = max(alpha * temperature, final_temperature)

        pso_sa_optimization()
        return global_best_position