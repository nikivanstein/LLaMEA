import numpy as np

class Enhanced_Hybrid_PSO_DE_Chaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.9
        self.crossover_rate = 0.9
        self.elite_fraction = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        # Modified initialization using chaotic map
        self.population = self.chaotic_initialization(self.pop_size, self.dim, self.lower_bound, self.upper_bound)
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.initial_pop_size = self.pop_size

    def chaotic_initialization(self, pop_size, dim, lower_bound, upper_bound):
        # Using a logistic map for chaotic initialization
        chaotic_seq = np.zeros((pop_size, dim))
        chaotic_seq[0, :] = np.random.uniform(0, 1, dim)
        for i in range(1, pop_size):
            chaotic_seq[i, :] = 4 * chaotic_seq[i-1, :] * (1 - chaotic_seq[i-1, :])
        return lower_bound + chaotic_seq * (upper_bound - lower_bound)

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return 0.01 * step

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.population)
            self.evaluations += self.pop_size

            better_scores_idx = scores < self.personal_best_scores
            self.personal_best_positions[better_scores_idx] = self.population[better_scores_idx]
            self.personal_best_scores[better_scores_idx] = scores[better_scores_idx]

            min_idx = np.argmin(scores)
            if scores[min_idx] < self.global_best_score:
                self.global_best_score = scores[min_idx]
                self.global_best_position = self.population[min_idx]

            # Adaptive inertia weight
            self.inertia_weight = 0.4 + 0.5 * (self.budget - self.evaluations) / self.budget
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.population)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.population)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            # Local search via DE for improved trial vectors (elite only)
            elite_count = int(self.elite_fraction * self.pop_size)
            elite_indices = np.argsort(scores)[:elite_count]
            for i in elite_indices:
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_score = func(trial_vector)
                self.evaluations += 1
                if trial_score < scores[i]:
                    self.population[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_positions[i] = trial_vector
                        self.personal_best_scores[i] = trial_score
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial_vector

            if np.random.rand() < 0.15:
                step = self.levy_flight((self.pop_size, self.dim))
                self.population += step
                self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            # Chaotic reinitialization for potential stagnation
            if self.evaluations % (self.budget // 10) == 0:
                self.population = self.chaotic_initialization(self.pop_size, self.dim, self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score