import numpy as np

class QiPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 15 * dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        # Initialize positions and velocities
        positions = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        fitness = np.apply_along_axis(func, 1, positions)
        self.evaluations = self.population_size
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))

                # Update position
                quantum_jump = np.random.rand(self.dim) < 0.05
                new_position = np.where(quantum_jump,
                                        self.lower_bound + np.random.rand(self.dim) * (self.upper_bound - self.lower_bound),
                                        positions[i] + velocities[i])
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = new_position
                    personal_best_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < global_best_fitness:
                        global_best_position = new_position
                        global_best_fitness = new_fitness

                # Update position
                positions[i] = new_position

        return global_best_position, global_best_fitness