import numpy as np

class EnhancedParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7  # Adjusted for more exploration
        self.cognitive_component = 1.5  # Slightly increased
        self.social_component = 2.0  # Slightly decreased for balance
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.adaptive_factor = 0.15  # Increased to enhance exploratory actions
        self.disturbance_factor = 0.1  # New factor for hybrid exploration

    def hybrid_exploration(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / np.abs(v) ** (1 / beta)

    def update_population_size(self):
        new_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
        if new_size < self.population_size:
            self.positions = self.positions[:new_size]
            self.velocities = self.velocities[:new_size]
            self.personal_best_positions = self.personal_best_positions[:new_size]
            self.personal_best_scores = self.personal_best_scores[:new_size]
            self.population_size = new_size

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

            inertia_weight = self.inertia_weight * (0.5 + 0.5 * np.cos(np.pi * self.evaluations / self.budget))

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Hybrid exploration with reinforced disturbances
            if np.random.rand() < self.adaptive_factor:
                for i in range(self.population_size):
                    disturbance = self.hybrid_exploration()
                    mutant = self.positions[i] + self.disturbance_factor * disturbance
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    mutant_fitness = func(mutant)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[i]:
                        self.positions[i] = mutant
                        self.personal_best_scores[i] = mutant_fitness
                        self.personal_best_positions[i] = mutant

            # Differential evolution inspired strategic mutation
            if np.random.rand() < self.adaptive_factor:
                for _ in range(int(self.population_size * 0.3)):  # Further increased mutation rate
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = self.positions[a] + 0.8 * (self.positions[b] - self.positions[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    candidate_index = np.random.randint(0, self.population_size)
                    mutant_fitness = func(mutant_vector)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[candidate_index]:
                        self.positions[candidate_index] = mutant_vector
                        self.personal_best_scores[candidate_index] = mutant_fitness
                        self.personal_best_positions[candidate_index] = mutant_vector

        return self.global_best_score