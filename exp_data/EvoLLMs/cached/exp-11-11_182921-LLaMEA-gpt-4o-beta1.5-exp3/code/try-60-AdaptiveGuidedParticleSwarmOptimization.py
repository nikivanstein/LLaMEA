import numpy as np

class AdaptiveGuidedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.func_evaluations = 0
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                current_score = func(self.population[i])
                self.func_evaluations += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.population[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.population[i]

            # Update velocities and positions
            for i in range(self.population_size):
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.population[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Adapt inertia weight
            self.inertia_weight = 0.9 - 0.5 * (self.func_evaluations / self.budget)

        return self.global_best_position