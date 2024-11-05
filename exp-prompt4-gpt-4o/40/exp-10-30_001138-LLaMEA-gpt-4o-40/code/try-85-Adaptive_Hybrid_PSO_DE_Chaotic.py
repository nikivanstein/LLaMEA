import numpy as np

class Adaptive_Hybrid_PSO_DE_Chaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 60  # Increased population size for diversity
        self.inertia_weight = 0.8  # Adjusted for better exploration-exploitation balance
        self.inertia_damping = 0.9  # Slightly higher damping for stability
        self.cognitive_coeff = 1.7  # Increased for improved local searching
        self.social_coeff = 1.4  # Fine-tuned for global exploration
        self.mutation_factor = 0.85  # Adjusted mutation factor
        self.crossover_rate = 0.9  # Higher crossover rate for exploitation
        self.elite_fraction = 0.15  # Adjusted elite fraction for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))  # Adjusted velocity range
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.chaotic_sequence = self.init_chaotic_sequence()

    def init_chaotic_sequence(self):
        x = np.random.rand()
        sequence = [x]
        for _ in range(self.budget * 2):
            x = 4 * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return 0.02 * step  # Adjusted step size for exploration

    def __call__(self, func):
        chaotic_idx = 0
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
            r1, r2 = self.chaotic_sequence[chaotic_idx:chaotic_idx+self.pop_size], self.chaotic_sequence[chaotic_idx+self.pop_size:chaotic_idx+2*self.pop_size]
            chaotic_idx += self.pop_size * 2
            cognitive_component = self.cognitive_coeff * r1[:, np.newaxis] * (self.personal_best_positions - self.population)
            social_component = self.social_coeff * r2[:, np.newaxis] * (self.global_best_position - self.population)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            if self.evaluations < self.budget * 0.7:
                elite_count = int(self.elite_fraction * self.pop_size)
                elite_indices = np.argsort(scores)[:elite_count]
                for i in range(self.pop_size):
                    if i not in elite_indices:
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

        return self.global_best_position, self.global_best_score