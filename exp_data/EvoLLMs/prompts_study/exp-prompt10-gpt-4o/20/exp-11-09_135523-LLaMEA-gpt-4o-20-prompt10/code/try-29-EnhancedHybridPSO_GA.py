import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.w = 0.6
        self.c1 = 1.8  # Slightly increased cognitive component for enhanced exploration
        self.c2 = 2.0  # Slightly reduced social component to maintain balance
        self.mutation_rate = 0.3  # Further increased mutation rate for higher diversity
        self.crossover_rate = 0.7  # Further increased crossover rate for improved exploration
        self.w_min = 0.3  # New parameter for dynamic inertia adjustment
        self.w_max = 0.9  # New parameter for dynamic inertia adjustment

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')

        secondary_best_position = np.zeros(self.dim)  # New secondary elite solution
        secondary_best_score = float('inf')

        eval_count = 0

        while eval_count < self.budget:
            scores = np.array([func(p) for p in positions])
            eval_count += self.num_particles

            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]

            # Update secondary best
            if personal_best_scores[min_score_idx] > global_best_score and personal_best_scores[min_score_idx] < secondary_best_score:
                secondary_best_score = personal_best_scores[min_score_idx]
                secondary_best_position = personal_best_positions[min_score_idx]

            adaptive_c1 = self.c1 * np.exp(-eval_count / (0.3 * self.budget))
            adaptive_c2 = self.c2 * (1 - np.exp(-eval_count / (0.3 * self.budget)))

            # Dynamic inertia adjustment
            self.w = self.w_min + (self.w_max - self.w_min) * (1 - eval_count / self.budget)

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (self.w * velocities +
                          adaptive_c1 * r1 * (personal_best_positions - positions) +
                          adaptive_c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            for i in range(self.num_particles):
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    partner = positions[np.random.choice(idxs)]
                    mask = np.random.rand(self.dim) < 0.5
                    child = np.where(mask, positions[i], partner)

                    if np.random.rand() < self.mutation_rate:
                        mutation_idx = np.random.randint(self.dim)
                        mutation_value = np.random.uniform(self.lb, self.ub)
                        child[mutation_idx] = mutation_value

                    child_score = func(child)
                    eval_count += 1
                    if child_score < scores[i]:
                        positions[i] = child
                        scores[i] = child_score

                    if eval_count >= self.budget:
                        break

            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

            # Use secondary elite for additional guidance
            if scores[current_best_idx] > global_best_score and scores[current_best_idx] < secondary_best_score:
                secondary_best_score = scores[current_best_idx]
                secondary_best_position = positions[current_best_idx]

            if eval_count >= self.budget:
                break

        return global_best_position