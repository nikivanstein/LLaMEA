import numpy as np

class APSO_DIWLL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4

    def __call__(self, func):
        particles = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
        fitness = np.apply_along_axis(func, 1, particles)
        self.evaluations = self.population_size

        personal_best_positions = np.copy(particles)
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocities[i] = (inertia_weight * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                                 + self.c2 * r2 * (global_best_position - particles[i]))

                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                current_fitness = func(particles[i])
                self.evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_fitness[i] = current_fitness

                if current_fitness < global_best_fitness:
                    global_best_position = particles[i]
                    global_best_fitness = current_fitness

        return global_best_position, global_best_fitness