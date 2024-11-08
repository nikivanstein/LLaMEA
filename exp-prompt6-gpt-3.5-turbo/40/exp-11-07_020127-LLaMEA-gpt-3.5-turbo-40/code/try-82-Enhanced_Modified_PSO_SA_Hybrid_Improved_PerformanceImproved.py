import numpy as np

class Enhanced_Modified_PSO_SA_Hybrid_Improved_PerformanceImproved:
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
            inertia_weight, cognitive_param, social_param = 0.5, 2.0, 2.0

            for _ in range(self.max_iter):
                fitness_values = np.array([func(p) for p in particles])

                update_mask = fitness_values < best_fitness
                best_fitness[update_mask], best_positions[update_mask] = fitness_values[update_mask], particles[update_mask]

                global_best_index = np.argmin(fitness_values)
                if fitness_values[global_best_index] < global_best_fitness:
                    global_best_fitness, global_best_position = fitness_values[global_best_index], particles[global_best_index]

                random_numbers = np.random.rand(self.num_particles, 2)
                inertia_cognitive = inertia_weight * cognitive_param
                inertia_social = inertia_weight * social_param
                diff_best = best_positions - particles
                diff_global = global_best_position - particles
                velocities = inertia_weight * velocities + inertia_cognitive * random_numbers[:, 0][:, None] * diff_best + inertia_social * random_numbers[:, 1][:, None] * diff_global
                particles = np.clip(particles + velocities, -5.0, 5.0)

                random_displacements = np.random.normal(0, 1, (self.num_particles, self.dim))
                new_positions = particles + random_displacements
                new_fitness_values = np.array([func(p) for p in new_positions])

                accept_mask = new_fitness_values < fitness_values
                accept_probabilities = np.exp((fitness_values - new_fitness_values) / temperature)
                accept_mask = np.logical_or(accept_mask, np.random.rand(self.num_particles) < accept_probabilities)

                particles = np.where(accept_mask[:, None], new_positions, particles)
                temperature = max(alpha * temperature, final_temperature)

        pso_sa_optimization()
        return global_best_position
