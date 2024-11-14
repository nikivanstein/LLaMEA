import numpy as np

class HybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 60
        self.initial_inertia = 0.8
        self.final_inertia = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.neighborhood_coeff = 0.5
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.1

    def __call__(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evaluations = 0
        neighborhood_size = max(3, self.swarm_size // 8)

        while evaluations < self.budget:
            inertia = self.initial_inertia - (self.initial_inertia - self.final_inertia) * (evaluations / self.budget)

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

            # Genetic algorithm components
            if np.random.rand() < self.crossover_prob:
                parents = np.random.choice(self.swarm_size, 2, replace=False)
                crossover_point = np.random.randint(1, self.dim)
                child1 = np.concatenate((position[parents[0], :crossover_point], position[parents[1], crossover_point:]))
                child2 = np.concatenate((position[parents[1], :crossover_point], position[parents[0], crossover_point:]))
                position[parents[0]] = child1
                position[parents[1]] = child2

            if np.random.rand() < self.mutation_prob:
                mutate_idx = np.random.choice(self.swarm_size)
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                position[mutate_idx] += mutation_vector
                position[mutate_idx] = np.clip(position[mutate_idx], self.lower_bound, self.upper_bound)

            adaptive_learning_rate = 0.5 + 0.5 * np.tanh(1.0 - evaluations / (self.budget * 0.9))

            for i in range(self.swarm_size):
                neighborhood_indices = np.random.choice(self.swarm_size, neighborhood_size, replace=False)
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