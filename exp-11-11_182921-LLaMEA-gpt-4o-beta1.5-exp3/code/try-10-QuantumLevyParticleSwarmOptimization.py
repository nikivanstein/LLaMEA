import numpy as np

class QuantumLevyParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.zeros((self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.inertia_weight = 0.9
        self.cognitive_param = 2.0
        self.social_param = 2.0

    def levy_flight(self, L):
        # Levy exponent and coefficient
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v)**(1 / beta)
        return step * L

    def __call__(self, func):
        # Evaluate initial population
        for i in range(self.population_size):
            score = func(self.population[i])
            self.func_evaluations += 1
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]
            if score < self.best_score:
                self.best_score = score
                self.best_position = self.population[i]

        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (self.personal_best_positions[i] - self.population[i])
                social_velocity = self.social_param * r2 * (self.best_position - self.population[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)

                # Update position
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate the new position
                score = func(self.population[i])
                self.func_evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = self.population[i]

                # Apply Levy flight for local search
                if np.random.rand() < 0.1:
                    levy_step = self.levy_flight(0.1)
                    trial_position = np.clip(self.population[i] + levy_step, self.lower_bound, self.upper_bound)
                    trial_score = func(trial_position)
                    self.func_evaluations += 1
                    if trial_score < score:
                        self.population[i] = trial_position
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_position
                        if trial_score < self.best_score:
                            self.best_score = trial_score
                            self.best_position = trial_position

            # Adapt inertia weight
            self.inertia_weight = 0.4 + 0.5 * (self.budget - self.func_evaluations) / self.budget

        return self.best_position