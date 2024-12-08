import numpy as np

class APSO_QDPR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.base_pop_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5

    def __call__(self, func):
        population_size = self.base_pop_size
        particles = self.lower_bound + np.random.rand(population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(population_size, self.dim) * 0.1
        personal_best_positions = np.copy(particles)
        personal_best_fitness = np.apply_along_axis(func, 1, particles)
        self.evaluations = population_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_const * r2 * (global_best_position - particles[i]))

                particles[i] += velocities[i]

                # Quantum-inspired dynamic
                quantum_move = np.random.rand(self.dim)
                quantum_mask = quantum_move < 0.1
                particles[i][quantum_mask] = self.lower_bound + np.random.rand(np.sum(quantum_mask)) * (self.upper_bound - self.lower_bound)

                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                fitness = func(particles[i])
                self.evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_fitness[i] = fitness

                    if fitness < global_best_fitness:
                        global_best_position = particles[i]
                        global_best_fitness = fitness

            # Dynamic population resizing
            if self.evaluations % (self.base_pop_size // 2) == 0:
                population_size = max(4, int(population_size * 0.9))
                indices = np.argsort(personal_best_fitness)[:population_size]
                particles = particles[indices]
                velocities = velocities[indices]
                personal_best_positions = personal_best_positions[indices]
                personal_best_fitness = personal_best_fitness[indices]

        return global_best_position, global_best_fitness