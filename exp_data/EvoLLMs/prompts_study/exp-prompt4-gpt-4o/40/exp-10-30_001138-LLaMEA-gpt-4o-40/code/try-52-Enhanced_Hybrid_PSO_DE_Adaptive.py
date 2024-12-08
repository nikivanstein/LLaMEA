import numpy as np

class Enhanced_Hybrid_PSO_DE_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.9
        self.inertia_damping = 0.98  # Adjusted damping for gradual decrease
        self.cognitive_coeff = 1.4  # Slightly reduced to balance exploration
        self.social_coeff = 1.6  # Slightly increased to enhance convergence
        self.mutation_factor = 0.8  # Adjusted for better diversification
        self.crossover_rate = 0.85  # Relaxed crossover for more diversity
        self.elite_fraction = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.chaos_factor = np.random.rand()  # Chaos factor for dynamic adjustments

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return 0.01 * step * self.chaos_factor  # Applied chaos factor

    def __call__(self, func):
        stagnation_counter = 0  # To monitor stagnation
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.population)
            self.evaluations += self.pop_size

            better_scores_idx = scores < self.personal_best_scores
            self.personal_best_positions[better_scores_idx] = self.population[better_scores_idx]
            self.personal_best_scores[better_scores_idx] = scores[better_scores_idx]

            min_idx = np.argmin(scores)
            if scores[min_idx] < self.global_best_score:
                stagnation_counter = 0  # Reset on improvement
                self.global_best_score = scores[min_idx]
                self.global_best_position = self.population[min_idx]
            else:
                stagnation_counter += 1  # Increment on stagnation

            self.inertia_weight *= self.inertia_damping + 0.01 * self.chaos_factor  # Dynamic adjustment
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
                adaptive_mutation = self.mutation_factor * (1 + np.random.uniform(-0.1, 0.1))  # Adaptive mutation
                mutant_vector = np.clip(a + adaptive_mutation * (b - c), self.lower_bound, self.upper_bound)
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

            if stagnation_counter > self.pop_size // 2 and np.random.rand() < 0.1:
                self.population += self.levy_flight((self.pop_size, self.dim))
                self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score