import numpy as np

class PSO_AMDI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 20 * dim
        self.initial_inertia = 0.9
        self.final_inertia = 0.4
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0

    def __call__(self, func):
        # Initialize positions and velocities
        positions = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.apply_along_axis(func, 1, positions)
        self.evaluations = self.swarm_size

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        while self.evaluations < self.budget:
            inertia_weight = self.initial_inertia - ((self.initial_inertia - self.final_inertia) * (self.evaluations / self.budget))
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity and position
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                current_score = func(positions[i])
                self.evaluations += 1

                # Update personal and global bests
                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = current_score

                if current_score < global_best_score:
                    global_best_position = positions[i]
                    global_best_score = current_score

        return global_best_position, global_best_score