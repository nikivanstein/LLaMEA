import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = max(5, 2 * self.dim)
        self.inertia_weight = 0.9
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.recombination_rate = 0.9
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.inf)

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                current_value = func(positions[i])
                self.evaluations += 1
                
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = positions[i]

                if current_value < self.best_global_value:
                    self.best_global_value = current_value
                    self.best_global_position = positions[i]

            self.inertia_weight = 0.9 - (0.5 * (self.evaluations / self.budget))

            for i in range(self.swarm_size):
                adaptive_cognitive = self.cognitive_coef * (1 - (self.evaluations / self.budget))
                adaptive_social = self.social_coef * (self.evaluations / self.budget)

                inertia = self.inertia_weight * velocities[i]
                cognitive = adaptive_cognitive * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social = adaptive_social * np.random.rand(self.dim) * (self.best_global_position - positions[i])
                velocities[i] = inertia + cognitive + social
                positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break
                a, b, c = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                mutant_vector = np.clip(personal_best_positions[a] + 
                                        self.mutation_factor * (personal_best_positions[b] - personal_best_positions[c]),
                                        self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.recombination_rate, mutant_vector, positions[i])
                
                trial_value = func(trial_vector)
                self.evaluations += 1
                
                if trial_value < personal_best_values[i]:
                    positions[i] = trial_vector
                    personal_best_values[i] = trial_value
                    personal_best_positions[i] = trial_vector

                    if trial_value < self.best_global_value:
                        self.best_global_value = trial_value
                        self.best_global_position = trial_vector

        return self.best_global_position