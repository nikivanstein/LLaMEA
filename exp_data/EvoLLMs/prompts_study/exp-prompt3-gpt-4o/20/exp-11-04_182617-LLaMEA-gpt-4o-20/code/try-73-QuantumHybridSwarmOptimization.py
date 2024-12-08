import numpy as np

class QuantumHybridSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30
        self.inertia = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.7
        self.velocity_scale = 0.1
        self.quantum_factor = 0.05  # New quantum factor for diversity boost
        self.mutation_rate = 0.1

    def __call__(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.velocity_scale, self.velocity_scale, (self.swarm_size, self.dim))

        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)

        global_best_value = np.inf
        global_best_position = None

        evaluations = 0

        while evaluations < self.budget:
            neighborhood_size = 5
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
                neighbors = np.random.choice(self.swarm_size, neighborhood_size, replace=False)
                local_best_position = personal_best_position[neighbors[np.argmin(personal_best_value[neighbors])]]
                
                r1, r2, r3 = np.random.rand(3)
                velocity[i] = (self.inertia * velocity[i] +
                               self.cognitive_coefficient * r1 * (personal_best_position[i] - position[i]) +
                               self.social_coefficient * r2 * (local_best_position - position[i]) +
                               self.quantum_factor * (np.random.uniform(self.lower_bound, self.upper_bound, self.dim) - position[i])
                               + np.std(position, axis=0).mean() * r3 * (global_best_position - position[i]))

                if np.random.rand() < self.mutation_rate:
                    velocity[i] *= np.random.uniform(-1.5, 1.5)

                position[i] += velocity[i]

                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

            self.inertia = 0.4 + 0.3 * (1 - evaluations / self.budget)

        return global_best_value