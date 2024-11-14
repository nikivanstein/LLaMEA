import numpy as np

class EnhancedAdaptiveNeighborhoodPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_swarm_size = 40  # Dynamic swarm size
        self.final_swarm_size = 80
        self.initial_inertia = 0.9  # Slightly increased initial inertia for broader exploration
        self.final_inertia = 0.4
        self.cognitive_coeff = 1.3  # Adjusted coefficients
        self.social_coeff = 1.8
        self.neighborhood_coeff = 0.6  # Increased neighborhood influence
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.15  # Increased max velocity

    def __call__(self, func):
        np.random.seed(42)
        swarm_size = self.initial_swarm_size
        position = np.random.uniform(self.lower_bound, self.upper_bound, (swarm_size, self.dim))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evaluations = 0
        neighborhood_size = max(3, swarm_size // 8)

        while evaluations < self.budget:
            inertia = self.initial_inertia - (self.initial_inertia - self.final_inertia) * (evaluations / self.budget)
            swarm_size = int(self.initial_swarm_size + (self.final_swarm_size - self.initial_swarm_size) * (evaluations / self.budget))
            position = np.resize(position, (swarm_size, self.dim))
            velocity = np.resize(velocity, (swarm_size, self.dim))
            personal_best_position = np.resize(personal_best_position, (swarm_size, self.dim))
            personal_best_value = np.resize(personal_best_value, swarm_size)

            for i in range(swarm_size):
                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i].copy()

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = position[i].copy()

                if evaluations >= self.budget:
                    break

            adaptive_learning_rate = 0.6 + 0.4 * np.tanh(1.0 - evaluations / (self.budget * 0.9))  # Slight change

            for i in range(swarm_size):
                neighborhood_indices = np.random.choice(swarm_size, neighborhood_size, replace=False)
                neighborhood_best = min(neighborhood_indices, key=lambda idx: personal_best_value[idx])
                neighborhood_best_position = personal_best_position[neighborhood_best]

                r1, r2, r3 = np.random.rand(self.dim), np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_term = inertia * velocity[i]
                cognitive_term = self.cognitive_coeff * r1 * (personal_best_position[i] - position[i])
                social_term = self.social_coeff * r2 * (global_best_position - position[i])
                neighborhood_term = self.neighborhood_coeff * r3 * (neighborhood_best_position - position[i])

                velocity[i] = adaptive_learning_rate * (inertia_term + cognitive_term + social_term + neighborhood_term)
                velocity[i] = np.clip(velocity[i], -self.max_velocity, self.max_velocity)
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

        return global_best_value