import numpy as np

class Enhanced_PSO_SA_Hybrid:
    def __init__(self, budget, dim, num_particles=30, max_iter=1000):
        self.budget, self.dim, self.num_particles, self.max_iter = budget, dim, num_particles, max_iter

    def __call__(self, func):
        def enhanced_pso_sa_optimization():
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
                update_indices = fitness_values < best_fitness
                
                best_fitness[update_indices] = fitness_values[update_indices]
                best_positions[update_indices] = particles[update_indices]

                global_best_index = np.argmin(fitness_values)
                if fitness_values[global_best_index] < global_best_fitness:
                    global_best_fitness, global_best_position = fitness_values[global_best_index], particles[global_best_index]

                r = np.random.rand(2, self.num_particles)
                velocities = 0.5 * velocities + 2.0 * r[0] * (best_positions - particles) + 2.0 * r[1] * (global_best_position - particles)
                particles = np.clip(particles + velocities, -5.0, 5.0)

                new_positions = particles + np.random.normal(0, 1, (self.num_particles, self.dim))
                new_fitness_values = func(new_positions)

                improvement_indices = new_fitness_values < fitness_values
                particles[improvement_indices] = new_positions[improvement_indices]

                acceptance_prob = np.exp((fitness_values - new_fitness_values) / temperature)
                random_probs = np.random.rand(self.num_particles)
                acceptance_indices = random_probs < acceptance_prob
                particles[acceptance_indices] = new_positions[acceptance_indices]

                temperature = np.maximum(alpha * temperature, final_temperature)

        enhanced_pso_sa_optimization()
        return global_best_position