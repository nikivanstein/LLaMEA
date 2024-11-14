import numpy as np

class PSO_AVC_DN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 10 * dim
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w = 0.5  # inertia weight
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)

    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitnesses = np.apply_along_axis(func, 1, positions)
        
        self.evaluations = self.pop_size

        # Identify the global best
        global_best_idx = np.argmin(personal_best_fitnesses)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitnesses[global_best_idx]

        # Main optimization loop
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))

                # Velocity clamping
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                fitness = func(positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < personal_best_fitnesses[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitnesses[i] = fitness

                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = fitness

            # Adaptively adjust inertia weight
            self.w = 0.4 + (0.9 - 0.4) * (self.budget - self.evaluations) / self.budget

        return global_best_position, global_best_fitness