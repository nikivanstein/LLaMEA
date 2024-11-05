import numpy as np

class EnhancedLearningParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30  # Increased population size for better exploration
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.3
        self.inertia_weight_max = 0.9
        self.cognitive_component = 1.5  # Adjusted for better balance
        self.social_component = 2.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def chaotic_map(self, iteration):
        return 0.7 * (1 - (iteration / self.budget)) * np.cos(10 * np.pi * iteration / self.budget)  # Changed function for diversity

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

            chaotic_factor = self.chaotic_map(self.evaluations)
            inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * 
                                                        (self.evaluations / self.budget)) ** 2  # Non-linear scaling

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            if np.random.rand() < 0.1:
                for i in range(self.population_size):
                    if np.random.rand() < 0.6:
                        crossover_point = np.random.randint(0, self.dim)
                        self.positions[i][:crossover_point] = chaotic_factor * self.global_best_position[:crossover_point] + \
                                                              (1 - chaotic_factor) * self.positions[i][:crossover_point]

        return self.global_best_score