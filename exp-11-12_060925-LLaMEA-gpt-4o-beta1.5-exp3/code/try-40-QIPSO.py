import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 30 * dim
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7   # inertia weight

    def __call__(self, func):
        # Initialize positions and velocities
        positions = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.pop_size, self.dim) - 0.5
        fitness = np.apply_along_axis(func, 1, positions)
        self.evaluations = self.pop_size

        # Initialize best positions
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity and position using QIPSO principles
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - positions[i])
                                 + self.c2 * r2 * (global_best_position - positions[i]))

                # Quantum-inspired update
                quantum_factor = np.random.uniform(-1, 1, self.dim)
                quantum_contribution = np.sin(quantum_factor * np.pi) * (global_best_position - positions[i])
                positions[i] += velocities[i] + quantum_contribution
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new fitness
                new_fitness = func(positions[i])
                self.evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = new_fitness

                if new_fitness < global_best_fitness:
                    global_best_position = positions[i]
                    global_best_fitness = new_fitness

        return global_best_position, global_best_fitness