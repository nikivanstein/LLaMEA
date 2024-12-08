import numpy as np

class EnhancedHybridPSO_GA_VNS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 35  # Slightly reduced to focus evaluations
        self.w = 0.5  # Reduced inertia for quicker convergence
        self.c1 = 1.4  # Fine-tuned cognitive component
        self.c2 = 2.5  # Further increased social component for global exploration
        self.mutation_rate = 0.3  # Higher mutation rate for maintaining diversity
        self.crossover_rate = 0.7  # Increased crossover rate for better exploitation

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

            adaptive_c1 = self.c1 * np.exp(-eval_count / (0.2 * self.budget))
            adaptive_c2 = self.c2 * (1 - np.exp(-eval_count / (0.2 * self.budget)))

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
                        mutation_step = np.random.uniform(-0.5, 0.5)
                        child[mutation_idx] += mutation_step
                        child = np.clip(child, self.lb, self.ub)

                    child_score = func(child)
                    eval_count += 1
                    if child_score < scores[i]:
                        positions[i] = child
                        scores[i] = child_score

                    # Apply Variable Neighborhood Search
                    if np.random.rand() < 0.1:
                        neighbor = child + np.random.normal(0, 0.1, self.dim)
                        neighbor = np.clip(neighbor, self.lb, self.ub)
                        neighbor_score = func(neighbor)
                        eval_count += 1
                        if neighbor_score < child_score:
                            positions[i] = neighbor
                            scores[i] = neighbor_score

                    if eval_count >= self.budget:
                        break

            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

        return global_best_position