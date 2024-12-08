import numpy as np

class EnhancedDynamicSwarmIntelligence:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30
        self.inertia = 0.7
        self.cognitive_coefficient = 1.7
        self.social_coefficient = 1.9
        self.velocity_scale = 0.1
        self.mutation_rate = 0.15

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        position = np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound) + self.lower_bound
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
                phi = self.cognitive_coefficient + self.social_coefficient
                constriction_factor = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))
                velocity[i] = (constriction_factor * (self.inertia * velocity[i] +
                               self.cognitive_coefficient * r1 * (personal_best_position[i] - position[i]) +
                               self.social_coefficient * r2 * (global_best_position - position[i])))

                if np.random.rand() < self.mutation_rate:
                    velocity[i] *= np.random.uniform(-2, 2)

                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

            self.inertia = 0.4 + 0.3 * ((np.cos(np.pi * evaluations / self.budget)) ** 2)

        return global_best_value