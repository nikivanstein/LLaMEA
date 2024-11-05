import numpy as np

class DynamicSwarmIntelligence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30
        self.inertia = 0.7  # Adjusted initial inertia for balance
        self.cognitive_coefficient = 1.7  # Slightly increased for faster personal learning
        self.social_coefficient = 1.9  # Adjusted social influence
        self.velocity_scale = 0.1
        self.mutation_rate = 0.15  # Slightly adjusted mutation rate

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.velocity_scale, self.velocity_scale, (self.swarm_size, self.dim))

        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)
        
        global_best_value = np.inf
        global_best_position = None

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i]

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = position[i]

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia * velocity[i] +
                               self.cognitive_coefficient * r1 * (personal_best_position[i] - position[i]) +
                               self.social_coefficient * r2 * (global_best_position - position[i]))

                # Chaotic mapping-based mutation for exploration
                if np.random.rand() < self.mutation_rate:
                    velocity[i] *= np.random.uniform(-2, 2)  # Increased scale for exploration

                position[i] += velocity[i]

                # Ensure the particles remain within bounds
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

            # Adaptive inertia weight adjustment
            self.inertia = 0.4 + 0.3 * ((np.cos(np.pi * evaluations / self.budget)) ** 2)

        return global_best_value