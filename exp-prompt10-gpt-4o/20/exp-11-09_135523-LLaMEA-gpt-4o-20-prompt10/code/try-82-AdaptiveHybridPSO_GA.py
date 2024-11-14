import numpy as np

class AdaptiveHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 35  # Reduced to concentrate on elite regions
        self.w = 0.7  # Adaptive inertia for dynamic balance
        self.c1 = 1.7  # Sharpened cognitive component
        self.c2 = 2.0  # Moderated social component for convergence control
        self.mutation_rate = 0.2  # Slightly reduced mutation rate
        self.crossover_rate = 0.7  # Raised crossover rate for better search

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')

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

            adaptive_w = self.w * (0.5 * (1 + np.tanh(10 * (0.5 - eval_count / self.budget))))
            adaptive_c1 = self.c1 * (1 - np.exp(-5 * eval_count / self.budget))
            adaptive_c2 = self.c2 * np.exp(-5 * (1 - eval_count / self.budget))

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (adaptive_w * velocities +
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
                        mutation_value = global_best_position[mutation_idx] + \
                                         np.random.normal(0, 0.1)
                        child[mutation_idx] = np.clip(mutation_value, self.lb, self.ub)

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

        return global_best_position