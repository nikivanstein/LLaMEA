import numpy as np

class PSO_SA_Hybrid_Improved_Performance:
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
                fitness_values = func(particles)
                update_mask = fitness_values < best_fitness

                best_fitness[update_mask] = fitness_values[update_mask]
                best_positions[update_mask] = particles[update_mask]

                best_indices = np.argmin(fitness_values)
                if fitness_values[best_indices] < global_best_fitness:
                    global_best_fitness = fitness_values[best_indices]
                    global_best_position = particles[best_indices]

                r1_r2 = np.random.rand(2, self.num_particles, self.dim)
                velocities = 0.5 * velocities + 2.0 * r1_r2[0] * (best_positions - particles) + 2.0 * r1_r2[1] * (global_best_position - particles)
                particles = np.clip(particles + velocities, -5.0, 5.0)

                new_positions = particles + np.random.normal(0, 1, (self.num_particles, self.dim))
                new_fitness_values = func(new_positions)

                improvement_mask = new_fitness_values < fitness_values
                accept_probabilities = np.exp((fitness_values - new_fitness_values) / temperature)
                random_values = np.random.rand(self.num_particles)

                update_positions_mask = improvement_mask | (random_values < accept_probabilities)
                particles[update_positions_mask] = new_positions[update_positions_mask]

                temperature = max(alpha * temperature, final_temperature)

        pso_sa_optimization()
        return global_best_position