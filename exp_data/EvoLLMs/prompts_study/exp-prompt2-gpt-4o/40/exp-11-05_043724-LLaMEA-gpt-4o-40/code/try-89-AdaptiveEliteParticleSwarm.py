import numpy as np

class AdaptiveEliteParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.elite_fraction = 0.2  # Elite preservation fraction
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_component = 1.5
        self.social_component = 2.2
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return 0.01 * u / np.abs(v) ** (1 / beta)  # Adjusted step size

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                fitness = func(self.positions[i])
                self.evaluations += 1

                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            # Linear inertia weight decay
            inertia_weight = self.inertia_weight_initial - (
                (self.inertia_weight_initial - self.inertia_weight_final) * (self.evaluations / self.budget))

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Elite preservation mechanism
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(self.personal_best_scores)[:elite_size]

            # Enhanced exploration with dynamic Levy flights for non-elite particles
            for i in range(self.population_size):
                if i not in elite_indices:
                    levy_step = self.levy_flight(self.dim)
                    self.positions[i] += levy_step
                    self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_score