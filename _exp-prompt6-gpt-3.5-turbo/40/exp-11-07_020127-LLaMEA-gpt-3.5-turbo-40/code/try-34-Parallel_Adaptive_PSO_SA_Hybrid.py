import numpy as np
from multiprocessing import Pool

class Parallel_Adaptive_PSO_SA_Hybrid:
    def __init__(self, budget, dim, num_particles=30, max_iter=1000):
        self.budget, self.dim, self.num_particles, self.max_iter = budget, dim, num_particles, max_iter

    def __call__(self, func):
        def parallel_fitness_evaluation(particles):
            with Pool() as p:
                fitness_values = np.array(p.map(func, particles))
            return fitness_values

        def adaptive_pso_sa_optimization():
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
                fitness_values = parallel_fitness_evaluation(particles)

                update_mask = fitness_values < best_fitness
                best_fitness[update_mask], best_positions[update_mask] = fitness_values[update_mask], particles[update_mask]

                global_best_index = np.argmin(fitness_values)
                if fitness_values[global_best_index] < global_best_fitness:
                    global_best_fitness, global_best_position = fitness_values[global_best_index], particles[global_best_index]

                random_numbers = np.random.rand(self.num_particles, 2)
                velocities = inertia_weight * velocities + cognitive_param * random_numbers[:, 0][:, None] * (best_positions - particles) + social_param * random_numbers[:, 1][:, None] * (global_best_position - particles)
                particles = np.clip(particles + velocities, -5.0, 5.0)

                adaptive_inertia_weight = max(0.4, inertia_weight - 0.001)
                adaptive_temperature = max(alpha * temperature, final_temperature)

                particles += np.random.normal(0, 1, (self.num_particles, self.dim))
                particles = np.clip(particles, -5.0, 5.0)

        adaptive_pso_sa_optimization()
        return global_best_position