import numpy as np

class DPSO_AVI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.velocity_clamp = 0.5

    def __call__(self, func):
        # Initialize particles
        position = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocity = np.random.rand(self.population_size, self.dim) * self.velocity_clamp
        personal_best_position = np.copy(position)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_position)
        self.evaluations = self.population_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity and position
                velocity[i] = (
                    self.inertia_weight * velocity[i] +
                    self.cognitive_coeff * r1 * (personal_best_position[i] - position[i]) +
                    self.social_coeff * r2 * (global_best_position - position[i])
                )
                velocity[i] = np.clip(velocity[i], -self.velocity_clamp, self.velocity_clamp)
                position[i] = position[i] + velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

                # Evaluate new fitness
                fitness = func(position[i])
                self.evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_position[i] = position[i]
                    personal_best_fitness[i] = fitness

                # Update global best
                if fitness < global_best_fitness:
                    global_best_position = position[i]
                    global_best_fitness = fitness

            # Dynamically adjust inertia weight
            self.inertia_weight *= 0.99  # Gradually decrease inertia weight

        return global_best_position, global_best_fitness