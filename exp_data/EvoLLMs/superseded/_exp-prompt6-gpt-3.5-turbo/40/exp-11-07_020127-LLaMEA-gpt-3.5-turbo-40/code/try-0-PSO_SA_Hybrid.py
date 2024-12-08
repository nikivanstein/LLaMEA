import numpy as np

class PSO_SA_Hybrid:
    def __init__(self, budget, dim, num_particles=30, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter

    def __call__(self, func):
        def pso_sa_optimization():
            # PSO initialization
            particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
            velocities = np.zeros((self.num_particles, self.dim))
            best_positions = particles.copy()
            best_fitness = np.full(self.num_particles, np.inf)
            global_best_position = np.zeros(self.dim)
            global_best_fitness = np.inf

            # Simulated Annealing parameters
            initial_temperature = 100.0
            final_temperature = 0.1
            alpha = 0.99

            temperature = initial_temperature

            iter_count = 0
            while iter_count < self.max_iter and func.evaluations < self.budget:
                for i in range(self.num_particles):
                    fitness = func(particles[i])
                    if fitness < best_fitness[i]:
                        best_fitness[i] = fitness
                        best_positions[i] = particles[i]

                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particles[i]

                    # PSO update
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = 0.5 * velocities[i] + 2.0 * r1 * (best_positions[i] - particles[i]) + 2.0 * r2 * (global_best_position - particles[i])
                    particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)

                    # Simulated Annealing
                    new_position = particles[i] + np.random.normal(0, 1, self.dim)
                    new_fitness = func(new_position)
                    if new_fitness < fitness or np.random.rand() < np.exp((fitness - new_fitness) / temperature):
                        particles[i] = new_position

                temperature = max(alpha * temperature, final_temperature)
                iter_count += 1

        pso_sa_optimization()
        return global_best_position