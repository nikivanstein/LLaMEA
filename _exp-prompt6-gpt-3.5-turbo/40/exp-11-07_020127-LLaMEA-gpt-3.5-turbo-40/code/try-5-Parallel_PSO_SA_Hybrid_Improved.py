import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Parallel_PSO_SA_Hybrid_Improved:
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

            def update_particle(i):
                nonlocal global_best_fitness, global_best_position
                fitness = func(particles[i])

                if fitness < best_fitness[i]:
                    best_fitness[i] = fitness
                    best_positions[i] = particles[i]

                if fitness < global_best_fitness:
                    global_best_fitness, global_best_position = fitness, particles[i]

                r1, r2 = np.random.rand(2)
                velocities[i] = 0.5 * velocities[i] + 2.0 * r1 * (best_positions[i] - particles[i]) + 2.0 * r2 * (global_best_position - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)

                new_position = particles[i] + np.random.normal(0, 1, self.dim)
                new_fitness = func(new_position)

                if new_fitness < fitness or np.random.rand() < np.exp((fitness - new_fitness) / temperature):
                    particles[i] = new_position

            with ThreadPoolExecutor() as executor:
                for _ in range(self.max_iter):
                    executor.map(update_particle, range(self.num_particles))
                    temperature = max(alpha * temperature, final_temperature)

        pso_sa_optimization()
        return global_best_position