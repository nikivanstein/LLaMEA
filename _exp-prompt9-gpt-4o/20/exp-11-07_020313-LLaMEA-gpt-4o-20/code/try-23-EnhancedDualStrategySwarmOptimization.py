import numpy as np

class EnhancedDualStrategySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 35  # Increased swarm size for enhanced exploration
        self.alpha_cognitive = 0.25  # Adjusted cognitive component for better self-discovery
        self.alpha_social = 0.25  # Adjusted social component for global influence
        self.inertia_start = 0.9  # Enhanced starting inertia weight
        self.inertia_end = 0.4  # Enhanced ending inertia weight

    def __call__(self, func):
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        particle_velocities = np.random.uniform(-0.7, 0.7, (self.swarm_size, self.dim))  # Adjusted initial velocity range
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            inertia_weight = self.inertia_end + (self.inertia_start - self.inertia_end) * ((self.budget - evaluations) / self.budget)
            topology_matrix = np.random.binomial(1, 0.2, (self.swarm_size, self.swarm_size))
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                local_best_position = np.copy(global_best_position)
                for j in range(self.swarm_size):
                    if topology_matrix[i, j] and personal_best_values[j] < func(local_best_position):
                        local_best_position = personal_best_positions[j]
                rand_cognitive = np.random.rand(self.dim)
                rand_social = np.random.rand(self.dim)
                particle_velocities[i] = (inertia_weight * particle_velocities[i]
                                         + self.alpha_cognitive * rand_cognitive * (personal_best_positions[i] - particle_positions[i])
                                         + self.alpha_social * rand_social * (local_best_position - particle_positions[i]))
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)
                current_value = func(particle_positions[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = np.copy(particle_positions[i])

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = np.copy(particle_positions[i])

        return global_best_position, global_best_value