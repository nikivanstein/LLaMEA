import numpy as np

class EnhancedParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 40
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.adaptive_factor = 0.1
        self.fitness_landscape_factor = 0.05

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def fitness_landscape_analysis(self, func):
        sample_points = np.random.uniform(self.lower_bound, self.upper_bound, (10, self.dim))
        sample_values = np.array([func(point) for point in sample_points])
        diversity_measure = np.std(sample_values) / np.mean(sample_values)
        return diversity_measure

    def __call__(self, func):
        while self.evaluations < self.budget:
            diversity_measure = self.fitness_landscape_analysis(func)
            adaptive_mutation_factor = self.adaptive_factor * (1 + self.fitness_landscape_factor / (1 + diversity_measure))

            for i in range(self.initial_population_size):
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

            inertia_weight = 0.7 * (self.budget - self.evaluations) / self.budget
            for i in range(self.initial_population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = 1.5 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = 2.0 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Enhanced exploration with strategic Levy flights
            if np.random.rand() < adaptive_mutation_factor:
                for i in range(self.initial_population_size):
                    levy_step = self.levy_flight(self.dim)
                    mutant = self.positions[i] + 0.02 * levy_step
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    mutant_fitness = func(mutant)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[i]:
                        self.positions[i] = mutant
                        self.personal_best_scores[i] = mutant_fitness
                        self.personal_best_positions[i] = mutant

            # Adaptive differential evolution inspired strategic mutation
            if np.random.rand() < adaptive_mutation_factor:
                for _ in range(int(self.initial_population_size * 0.2)):
                    a, b, c = np.random.choice(self.initial_population_size, 3, replace=False)
                    mutant_vector = self.positions[a] + 0.6 * (self.positions[b] - self.positions[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    candidate_index = np.random.randint(0, self.initial_population_size)
                    mutant_fitness = func(mutant_vector)
                    self.evaluations += 1
                    if mutant_fitness < self.personal_best_scores[candidate_index]:
                        self.positions[candidate_index] = mutant_vector
                        self.personal_best_scores[candidate_index] = mutant_fitness
                        self.personal_best_positions[candidate_index] = mutant_vector

        return self.global_best_score