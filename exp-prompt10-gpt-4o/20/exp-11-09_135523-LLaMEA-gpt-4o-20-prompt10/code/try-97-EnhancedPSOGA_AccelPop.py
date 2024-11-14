import numpy as np

class EnhancedPSOGA_AccelPop:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 30  # Reduced initial particle count for faster initial convergence
        self.w = 0.6  # Lower inertia for quicker response to environmental changes
        self.c1 = 1.5  # Increased cognitive component for more aggressive personal best tracking
        self.c2 = 1.8  # Slightly reduced social component for balanced attraction
        self.mutation_rate = 0.35  # Increased mutation rate for enhanced exploration
        self.crossover_rate = 0.7  # Higher crossover rate for more frequent diversity
        self.resize_factor = 0.9  # Dynamic resizing of the population size to focus resources

    def __call__(self, func):
        # Initialize particles and auxiliary variables
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')

        eval_count = 0

        while eval_count < self.budget:
            # Evaluate particles
            scores = np.array([func(p) for p in positions])
            eval_count += len(positions)

            # Update personal and global bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]

            # Dynamic adjustments of cognitive and social components
            adaptive_c1 = self.c1 * (1 + np.sin(eval_count / self.budget * np.pi))
            adaptive_c2 = self.c2 * (1 + np.cos(eval_count / self.budget * np.pi))

            # Update velocities and positions using enhanced PSO-GA update rule
            r1 = np.random.rand(len(positions), self.dim)
            r2 = np.random.rand(len(positions), self.dim)
            velocities = (self.w * velocities +
                          adaptive_c1 * r1 * (personal_best_positions - positions) +
                          adaptive_c2 * r2 * (global_best_position - positions))
            velocities = np.clip(velocities, -0.4, 0.4)
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Genetic Algorithm operations with mutation strategy
            for i in range(len(positions)):
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(len(positions)) if idx != i]
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

            # Update global best with the latest evaluations
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

            # Dynamic population resizing
            if eval_count % (self.budget // 10) == 0 and eval_count < self.budget * 0.9:
                new_size = int(len(positions) * self.resize_factor)
                if new_size > 5:
                    sorted_indices = np.argsort(scores)
                    positions = positions[sorted_indices[:new_size]]
                    velocities = velocities[sorted_indices[:new_size]]
                    personal_best_positions = personal_best_positions[sorted_indices[:new_size]]
                    personal_best_scores = personal_best_scores[sorted_indices[:new_size]]

        return global_best_position