import numpy as np

class AdaptiveQuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 30
        self.inertia = 0.9
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 2.0
        self.velocity_scale = 0.1
        self.mutation_rate = 0.1  # Reduced mutation rate
        self.qbit_prob = 0.3  # New quantum behavior probability
        self.qfield_intensity = 0.5  # New parameter for quantum field intensity

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

                # Quantum-inspired influence
                if np.random.rand() < self.qbit_prob:
                    velocity[i] += self.qfield_intensity * (np.random.uniform(self.lower_bound, self.upper_bound, self.dim) - position[i])

                position[i] += velocity[i]

                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

            self.inertia = 0.4 + 0.5 * ((np.sin(np.pi * evaluations / self.budget)) ** 2)

        return global_best_value