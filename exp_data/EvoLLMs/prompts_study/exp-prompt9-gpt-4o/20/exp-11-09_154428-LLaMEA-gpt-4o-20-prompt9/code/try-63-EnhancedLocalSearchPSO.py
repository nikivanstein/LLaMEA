import numpy as np

class EnhancedLocalSearchPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 60
        self.initial_inertia = 0.9  # Slightly increased initial inertia
        self.final_inertia = 0.3  # Adjusted final inertia for stability
        self.cognitive_coeff = 1.5  # Further enhanced for local search
        self.social_coeff = 1.8  # Slightly reduced for finer exploration
        self.exploration_coeff = 0.7  # Further increased for exploration
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.15

    def __call__(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evaluations = 0

        while evaluations < self.budget:
            inertia = self.initial_inertia - (self.initial_inertia - self.final_inertia) * (evaluations / self.budget)

            elite = np.argmin(personal_best_value)
            elite_position = personal_best_position[elite]
            
            for i in range(self.swarm_size):
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

            adaptive_learning_rate = np.clip(1.0 - evaluations / (self.budget * 0.75), 0.3, 1.0)  # Adjusted learning rate bounds

            for i in range(self.swarm_size):
                r1, r2, r3 = np.random.rand(self.dim), np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_term = inertia * velocity[i]
                cognitive_term = self.cognitive_coeff * r1 * (personal_best_position[i] - position[i])
                social_term = self.social_coeff * r2 * (global_best_position - position[i])

                dynamic_neighbors = np.random.choice(self.swarm_size, 3, replace=False)
                exploration_term = self.exploration_coeff * np.mean(
                    [position[neighbor] - position[i] for neighbor in dynamic_neighbors], axis=0)

                elite_influence = 0.1 * r3 * (elite_position - position[i])  # Reduced elite influence

                velocity[i] = adaptive_learning_rate * (inertia_term + cognitive_term + social_term + exploration_term + elite_influence)
                velocity[i] += 0.05 * np.random.normal(size=self.dim)  # Stochastic perturbation added
                velocity[i] = np.clip(velocity[i], -self.max_velocity, self.max_velocity)
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

        return global_best_value