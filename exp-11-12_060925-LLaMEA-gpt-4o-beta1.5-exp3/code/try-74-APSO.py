import numpy as np

class APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 20 * dim
        self.inertia_weight = 0.9
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component

    def __call__(self, func):
        # Initialize particles
        positions = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
        personal_best_positions = positions.copy()
        personal_best_fitness = np.apply_along_axis(func, 1, positions)
        self.evaluations = self.swarm_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocities and positions
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                fitness = func(positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = fitness

            # Adapt inertia weight
            self.inertia_weight = 0.4 + 0.5 * (self.budget - self.evaluations) / self.budget

        return global_best_position, global_best_fitness