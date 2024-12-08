import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 50  # Reduced swarm size for faster exploration
        self.initial_inertia = 0.9  # Increased initial inertia for better exploration
        self.final_inertia = 0.3
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0  # Increased social influence
        self.exploration_coeff = 0.6  # Enhanced exploration coefficient
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.2  # Increased velocity range

    def __call__(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evaluations = 0
        subgroup_factor = 3  # Adjusted subgroup factor for better focus

        while evaluations < self.budget:
            inertia = self.final_inertia + (self.initial_inertia - self.final_inertia) * np.exp(-5 * (evaluations / self.budget))

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

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_term = inertia * velocity[i]
                cognitive_term = self.cognitive_coeff * r1 * (personal_best_position[i] - position[i])
                social_term = self.social_coeff * r2 * (global_best_position - position[i])

                donor_particles = np.random.choice(self.swarm_size, int(self.swarm_size / subgroup_factor), replace=False)
                exploration_term = self.exploration_coeff * np.mean(
                    [position[donor] - position[i] for donor in donor_particles], axis=0)

                # Enhanced leader influence
                leader_influence = 0.2 * np.random.rand(self.dim) * (elite_position - position[i])

                velocity[i] = inertia_term + cognitive_term + social_term + exploration_term + leader_influence
                velocity[i] = np.clip(velocity[i], -self.max_velocity, self.max_velocity)
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

        return global_best_value