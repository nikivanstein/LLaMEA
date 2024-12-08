import numpy as np

class EnhancedLevyAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 2.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / abs(v)**(1 / beta)
        return step

    def update_population_size(self):
        new_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
        if new_size < self.population_size:
            self.positions = self.positions[:new_size]
            self.velocities = self.velocities[:new_size]
            self.personal_best_positions = self.personal_best_positions[:new_size]
            self.personal_best_scores = self.personal_best_scores[:new_size]
            self.population_size = new_size

    def dynamic_learning(self):
        return 1.0 / (1 + np.exp(-(10 * self.evaluations / self.budget - 5)))

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.update_population_size()
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

            inertia_weight = self.inertia_weight * (self.budget - self.evaluations) / self.budget

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                dynamic_factor = self.dynamic_learning()
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity) * dynamic_factor
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            if np.random.rand() < 0.1:
                L = self.levy_flight(self.dim)
                for i in range(self.population_size):
                    mutant = self.positions[i] + L
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    mutant_fitness = func(mutant)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[i]:
                        self.positions[i] = mutant
                        self.personal_best_scores[i] = mutant_fitness
                        self.personal_best_positions[i] = mutant

        return self.global_best_score