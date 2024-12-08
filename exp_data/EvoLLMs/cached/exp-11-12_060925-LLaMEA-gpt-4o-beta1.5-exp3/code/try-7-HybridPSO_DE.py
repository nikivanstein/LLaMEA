import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.inertia = 0.5
        self.cognitive = 1.5
        self.social = 1.5
        self.f = 0.5  # DE scaling factor
        self.cr = 0.9  # DE crossover rate

    def __call__(self, func):
        # Initialize particle positions and velocities for PSO
        positions = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.swarm_size, self.dim) * 0.1
        personal_best_positions = np.copy(positions)
        personal_best_values = np.apply_along_axis(func, 1, personal_best_positions)
        self.evaluations = self.swarm_size

        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)

                # DE mutation and crossover
                indices = np.random.choice(self.swarm_size, 3, replace=False)
                x1, x2, x3 = positions[indices[0]], positions[indices[1]], positions[indices[2]]
                mutant_vector = np.clip(x1 + self.f * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, positions[i])

                # Evaluate positions
                fitness_pso = func(positions[i])
                fitness_de = func(trial_vector)
                self.evaluations += 2  # Two function calls

                # Choose the better method
                if fitness_pso < fitness_de:
                    new_position = positions[i]
                    new_fitness = fitness_pso
                else:
                    new_position = trial_vector
                    new_fitness = fitness_de

                # Update personal and global bests
                if new_fitness < personal_best_values[i]:
                    personal_best_positions[i] = new_position
                    personal_best_values[i] = new_fitness
                    if new_fitness < global_best_value:
                        global_best_position = new_position
                        global_best_value = new_fitness

                # Early stopping if budget is exhausted
                if self.evaluations >= self.budget:
                    break

        return global_best_position, global_best_value