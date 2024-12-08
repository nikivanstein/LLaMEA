import numpy as np

class Quantum_Hybrid_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.6  # Reduced for more dynamic adaptation
        self.inertia_damping = 0.94  # Slightly modified for stability
        self.cognitive_coeff = 1.8  # Slightly boosted for stronger local search
        self.social_coeff = 1.6  # Increased for better convergence
        self.mutation_factor = 0.85  # Enhanced for diversity
        self.crossover_rate = 0.8  # Slightly adjusted for balance
        self.elite_fraction = 0.1  # Reduced elite fraction for diverse search
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

    def quantum_update(self, size):
        theta = np.random.uniform(0, 2 * np.pi, size)
        r = np.random.rand(size)
        step = np.sqrt(-2 * np.log(r)) * np.cos(theta)
        return 0.02 * step  # Enhanced quantum step scale

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

            if self.evaluations > self.budget * 0.6:
                self.pop_size = max(15, int(self.initial_pop_size * (self.budget - self.evaluations) / self.budget))
                self.population = self.population[:self.pop_size]
                self.velocities = self.velocities[:self.pop_size]
                self.personal_best_positions = self.personal_best_positions[:self.pop_size]
                self.personal_best_scores = self.personal_best_scores[:self.pop_size]

            if np.random.rand() < 0.15:
                step = self.quantum_update((self.pop_size, self.dim))  # Integrating quantum steps
                self.population += step
                self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            if self.evaluations % (self.budget // 8) == 0:
                stagnant_indices = scores > np.median(scores)
                self.population[stagnant_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (stagnant_indices.sum(), self.dim))

        return self.global_best_position, self.global_best_score