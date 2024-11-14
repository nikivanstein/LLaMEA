import numpy as np

class MultiSwarmPSO:
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

    def __call__(self, func):
        # Initialize particle positions and velocities
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
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # Update velocity
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)
                
                # Evaluate the new position
                fitness = func(positions[i])
                self.evaluations += 1
                
                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = fitness
                    
                    # Update global best
                    if fitness < global_best_value:
                        global_best_position = positions[i]
                        global_best_value = fitness
                
                # Early stopping if budget is exhausted
                if self.evaluations >= self.budget:
                    break

        return global_best_position, global_best_value