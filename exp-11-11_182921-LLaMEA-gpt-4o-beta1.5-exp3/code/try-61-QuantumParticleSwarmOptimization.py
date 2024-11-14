import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_global_score = float('inf')
        self.best_global_position = None
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_scores = np.full(self.population_size, float('inf'))
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.tau = 0.05  # Quantum tunneling probability

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate current particle
                current_score = func(self.population[i])
                self.func_evaluations += 1
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.population[i]
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.population[i]
                
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_component * r1 * (self.best_personal_positions[i] - self.population[i])
                social_velocity = self.social_component * r2 * (self.best_global_position - self.population[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                
                # Quantum tunneling
                if np.random.rand() < self.tau:
                    self.population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                else:
                    self.population[i] += self.velocities[i]
                    self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            # Adaptive parameters
            self.inertia_weight = 0.9 - 0.5 * (self.func_evaluations / self.budget)
            self.tau = 0.05 * (1 - np.cos(2 * np.pi * self.func_evaluations / self.budget))

        return self.best_global_position