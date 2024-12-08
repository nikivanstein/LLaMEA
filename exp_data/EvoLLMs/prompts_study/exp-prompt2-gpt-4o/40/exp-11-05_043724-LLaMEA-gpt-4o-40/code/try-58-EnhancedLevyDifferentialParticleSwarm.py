import numpy as np

class EnhancedLevyDifferentialParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 40  # Increased population size
        self.population_size = self.initial_population_size
        self.inertia_weight_initial = 0.9  # Updated for exponential decay
        self.inertia_weight_final = 0.4
        self.cognitive_component = 1.5
        self.social_component = 2.5  # Adjusted for stronger social influence
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def exponential_decay_inertia(self):
        return self.inertia_weight_final + (self.inertia_weight_initial - self.inertia_weight_final) * \
               np.exp(-3 * self.evaluations / self.budget)

    def elite_guided_search(self):
        elite_positions = self.positions[np.argsort(self.personal_best_scores)[:5]]
        return elite_positions[np.random.randint(0, elite_positions.shape[0])]

    def bidirectional_learning(self, index):
        return (self.personal_best_positions[index] + self.global_best_position) * 0.5

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

            inertia_weight = self.exponential_decay_inertia()

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                elite_influence = 0.1 * (self.elite_guided_search() - self.positions[i])  # Elite-guided influence
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity + elite_influence)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Bidirectional learning for enhanced exploration
            for i in range(self.population_size):
                if np.random.rand() < 0.2:
                    bidirectional_step = self.bidirectional_learning(i)
                    self.positions[i] = np.clip(bidirectional_step, self.lower_bound, self.upper_bound)

        return self.global_best_score