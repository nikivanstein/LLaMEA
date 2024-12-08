import numpy as np

class SwarmDynamicsOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness
                score = func(self.population[i])
                self.func_evaluations += 1
                
                # Update personal bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                
                # Update global best
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.population[i]

            # Update velocity and position
            for i in range(self.population_size):
                inertia = self.inertia_weight * self.velocities[i]
                cognitive_component = self.cognitive_coeff * np.random.random(self.dim) * (self.personal_best_positions[i] - self.population[i])
                social_component = self.social_coeff * np.random.random(self.dim) * (self.best_position - self.population[i])
                
                self.velocities[i] = inertia + cognitive_component + social_component
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            # Adaptively adjust coefficients
            self.inertia_weight = 0.9 - 0.5 * (self.func_evaluations / self.budget)
            self.cognitive_coeff = 1.5 + 0.5 * np.sin(np.pi * self.func_evaluations / self.budget)
            self.social_coeff = 1.5 - 0.5 * np.sin(np.pi * self.func_evaluations / self.budget)

        return self.best_position