import numpy as np

class HPSO_AMCC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 10 * dim
        self.inertia = 0.729  # Constriction Coefficient
        self.c1 = 1.49445  # Cognitive Component
        self.c2 = 1.49445  # Social Component
        self.mutation_rate = 0.1

    def __call__(self, func):
        position = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocity = np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound) / 2.0
        personal_best_position = np.copy(position)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_position)
        self.evaluations += self.pop_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocity[i] = (
                    self.inertia * velocity[i]
                    + self.c1 * r1 * (personal_best_position[i] - position[i])
                    + self.c2 * r2 * (global_best_position - position[i])
                )
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

                # Adaptive mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = (np.random.rand(self.dim) - 0.5) * (self.upper_bound - self.lower_bound) * 0.1
                    position[i] += mutation_vector
                    position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

                current_fitness = func(position[i])
                self.evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_position[i] = position[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best_position = position[i]
                        global_best_fitness = current_fitness

        return global_best_position, global_best_fitness