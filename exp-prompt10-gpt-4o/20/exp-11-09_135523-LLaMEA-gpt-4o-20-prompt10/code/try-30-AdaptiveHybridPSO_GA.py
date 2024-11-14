import numpy as np

class AdaptiveHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 45  # Slightly increased particles for enhanced diversity
        self.w = 0.5  # Lower inertia for quicker convergence
        self.c1 = 1.4  # Dynamic cognitive component
        self.c2 = 2.3  # Further increased social component
        self.mutation_rate = 0.3  # Elevated mutation rate for exploration
        self.crossover_rate = 0.65  # Optimized crossover rate for robust search
        self.local_search_prob = 0.1  # Probability of performing local search

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.2, 0.2, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')

        eval_count = 0

        while eval_count < self.budget:
            # Evaluate particles
            scores = np.array([func(p) for p in positions])
            eval_count += self.num_particles

            # Update personal and global bests
            better_mask = scores < personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]

            # Adaptive cognitive and social components with chaotic perturbation
            adaptive_c1 = self.c1 * (1 - np.sin(0.1 * eval_count))
            adaptive_c2 = self.c2 * (0.5 + 0.5 * np.cos(0.1 * eval_count))

            # Update velocities and positions using hybrid PSO-GA update rule
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (self.w * velocities +
                          adaptive_c1 * r1 * (personal_best_positions - positions) +
                          adaptive_c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Genetic Algorithm operations
            for i in range(self.num_particles):
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    partner = positions[np.random.choice(idxs)]
                    mask = np.random.rand(self.dim) < 0.6
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

            # Local search for fine-tuning
            if np.random.rand() < self.local_search_prob and eval_count < self.budget:
                for j in range(self.dim):
                    perturbed_position = np.copy(global_best_position)
                    perturbation = np.random.uniform(-0.1, 0.1)
                    perturbed_position[j] += perturbation
                    perturbed_position = np.clip(perturbed_position, self.lb, self.ub)
                    perturbed_score = func(perturbed_position)
                    eval_count += 1
                    if perturbed_score < global_best_score:
                        global_best_score = perturbed_score
                        global_best_position = perturbed_position

            # Update global best with the latest evaluations
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

        return global_best_position