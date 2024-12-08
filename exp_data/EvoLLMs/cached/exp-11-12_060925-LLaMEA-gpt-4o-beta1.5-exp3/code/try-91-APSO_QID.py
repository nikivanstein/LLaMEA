import numpy as np

class APSO_QID:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.inertia_weight = 0.9
        self.cognitive_const = 2.0
        self.social_const = 2.0
        self.qubit_prob = 0.2

    def __call__(self, func):
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_positions)
        self.evaluations = self.population_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocities with adaptive inertia
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_const * r2 * (global_best_position - population[i]))

                # Quantum-inspired position update
                if np.random.rand() < self.qubit_prob:
                    indices = np.random.permutation(self.population_size)
                    x1, x2 = population[indices[:2]]
                    direction = np.random.choice([-1, 1], self.dim)
                    new_position = global_best_position + direction * np.abs(x1 - x2)
                else:
                    new_position = population[i] + velocities[i]

                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = new_position
                    personal_best_fitness[i] = new_fitness

                    if new_fitness < global_best_fitness:
                        global_best_position = new_position
                        global_best_fitness = new_fitness

            # Adaptive inertia weight decay
            self.inertia_weight = max(0.4, self.inertia_weight * 0.99)

        return global_best_position, global_best_fitness