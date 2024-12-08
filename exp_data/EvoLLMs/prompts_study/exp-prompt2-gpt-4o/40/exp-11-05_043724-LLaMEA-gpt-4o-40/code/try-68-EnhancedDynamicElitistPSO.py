import numpy as np

class EnhancedDynamicElitistPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 40
        self.population_size = self.initial_population_size
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def adaptive_inertia_weight(self):
        return self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (self.evaluations / self.budget)

    def update_positions_and_velocities(self, func):
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

        inertia_weight = self.adaptive_inertia_weight()

        for i in range(self.population_size):
            r1 = np.random.uniform(0, 1, self.dim)
            r2 = np.random.uniform(0, 1, self.dim)
            cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = (inertia_weight * self.velocities[i] +
                                  cognitive_velocity + social_velocity)
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

    def elitism_crossover(self, func):
        elite_size = int(0.2 * self.population_size)
        elite_indices = np.argsort(self.personal_best_scores)[:elite_size]
        for i in range(elite_size):
            for _ in range(2):
                mate_index = np.random.choice(elite_indices)
                crossover_vector = (self.personal_best_positions[elite_indices[i]] + 
                                    self.personal_best_positions[mate_index]) / 2
                crossover_vector = np.clip(crossover_vector, self.lower_bound, self.upper_bound)
                crossover_fitness = func(crossover_vector)
                self.evaluations += 1
                if crossover_fitness < self.personal_best_scores[elite_indices[i]]:
                    self.personal_best_scores[elite_indices[i]] = crossover_fitness
                    self.personal_best_positions[elite_indices[i]] = crossover_vector

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.update_positions_and_velocities(func)
            if np.random.rand() < 0.2:
                self.elitism_crossover(func)

        return self.global_best_score