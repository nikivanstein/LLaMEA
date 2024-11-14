import numpy as np

class AdaptiveSwarmQuantumMutation:
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
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.tau = 0.1  # Initial quantum mutation probability

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocities
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.cognitive_coeff * r1 * (self.personal_best[i] - self.population[i]) +
                                      self.social_coeff * r2 * (self.best_position - self.population[i]))

                # Apply velocity and clip
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Quantum mutation
                if np.random.rand() < self.tau:
                    self.population[i] += np.random.normal(0, 1, self.dim)
                    self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                score = func(self.population[i])
                self.func_evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best[i] = self.population[i]
                    self.personal_best_scores[i] = score

                # Update global best
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.population[i]

            # Adaptive adjustment of inertia weight and tau
            self.inertia_weight = 0.9 - 0.5 * (self.func_evaluations / self.budget)
            self.tau = 0.1 * (1 + np.sin(2 * np.pi * self.func_evaluations / self.budget))

        return self.best_position