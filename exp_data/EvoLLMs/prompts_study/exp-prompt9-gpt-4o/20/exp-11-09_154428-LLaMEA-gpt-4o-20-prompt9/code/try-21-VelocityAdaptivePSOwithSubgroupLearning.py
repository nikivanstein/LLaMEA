import numpy as np

class VelocityAdaptivePSOwithSubgroupLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 50
        self.inertia = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.differential_coeff = 0.6  # Slightly increase differential influence
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.15  # Increase max velocity

    def __call__(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evaluations = 0
        subgroup_size = max(1, self.swarm_size // 5)

        while evaluations < self.budget:
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

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_term = self.inertia * velocity[i]
                cognitive_term = self.cognitive_coeff * r1 * (personal_best_position[i] - position[i])
                social_term = self.social_coeff * r2 * (global_best_position - position[i])
                
                donor_particles = np.random.choice(self.swarm_size, subgroup_size, replace=False)
                differential_term = self.differential_coeff * np.mean(
                    [position[donor] - position[i] for donor in donor_particles], axis=0)

                velocity[i] = inertia_term + cognitive_term + social_term + differential_term
                velocity[i] = np.clip(velocity[i], -self.max_velocity, self.max_velocity)
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

        return global_best_value