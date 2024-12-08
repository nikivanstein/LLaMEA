import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 20
        self.alpha = 0.05

    def __call__(self, func):
        particle_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        particle_velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particle_positions)
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                particle_velocities[i] = 0.7 * particle_velocities[i] + self.alpha * np.random.rand(self.dim) * (personal_best_positions[i] - particle_positions[i]) + self.alpha * np.random.rand(self.dim) * (global_best_position - particle_positions[i])
                particle_positions[i] += particle_velocities[i]
                particle_positions[i] = np.clip(particle_positions[i], self.lower_bound, self.upper_bound)
                current_value = func(particle_positions[i])
                evaluations += 1
                
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particle_positions[i]
                    
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particle_positions[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_value