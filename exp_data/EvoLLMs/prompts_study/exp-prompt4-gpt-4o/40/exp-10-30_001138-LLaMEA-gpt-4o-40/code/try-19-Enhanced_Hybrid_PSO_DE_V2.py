import numpy as np

class Enhanced_Hybrid_PSO_DE_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.9
        self.inertia_damping = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.9
        self.crossover_rate = 0.9
        self.elite_fraction = 0.2  # Increased elite fraction for adaptive selection
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.initial_pop_size = self.pop_size

    def chaotic_mapping(self, size):
        # Logistic map for chaotic sequence
        x = np.random.rand(size)
        r = 4.0  # Chaotic parameter
        return r * x * (1 - x)

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

            self.inertia_weight *= self.inertia_damping
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.population)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.population)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            elite_count = int(self.elite_fraction * self.pop_size)
            elite_indices = np.argsort(scores)[:elite_count]
            for i in range(self.pop_size):
                if i in elite_indices:
                    continue
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutation_factor_dynamic = self.mutation_factor * (1 - (self.evaluations / self.budget))  # Dynamic mutation
                mutant_vector = np.clip(a + mutation_factor_dynamic * (b - c), self.lower_bound, self.upper_bound)
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
            
            if self.evaluations > self.budget * 0.5:
                self.pop_size = max(10, int(self.initial_pop_size * (self.budget - self.evaluations) / self.budget))
                self.population = self.population[:self.pop_size]
                self.velocities = self.velocities[:self.pop_size]
                self.personal_best_positions = self.personal_best_positions[:self.pop_size]
                self.personal_best_scores = self.personal_best_scores[:self.pop_size]

            if np.random.rand() < 0.15:
                step = self.levy_flight((self.pop_size, self.dim))
                self.population += step
                self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            if self.evaluations % (self.budget // 10) == 0:
                stagnant_indices = scores > np.mean(scores)
                self.population[stagnant_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (stagnant_indices.sum(), self.dim))
                self.population += self.chaotic_mapping(self.dim)  # Add chaos-induced diversity

        return self.global_best_position, self.global_best_score