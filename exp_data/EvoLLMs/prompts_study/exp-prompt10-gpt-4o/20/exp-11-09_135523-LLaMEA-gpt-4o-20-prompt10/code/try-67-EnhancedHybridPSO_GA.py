import numpy as np

class EnhancedHybridPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.w = 0.5  # Increased inertia to enhance exploration
        self.c1 = 1.7  # Increased cognitive component for personal best attraction
        self.c2 = 2.0  # Adjusted social component for balance
        self.mutation_rate = 0.3  # Further increased mutation for diversity
        self.crossover_rate = 0.7  # Enhanced crossover rate for exploration
        self.memory_decay = 0.9  # Decay factor for adaptive memory

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')
        adaptive_memory = np.zeros(self.dim)  # Added adaptive memory

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

            # Update adaptive memory
            adaptive_memory = self.memory_decay * adaptive_memory + (1 - self.memory_decay) * global_best_position

            # Adaptive cognitive and social components with memory influence
            adaptive_c1 = self.c1 * (1 - eval_count / self.budget)
            adaptive_c2 = self.c2 * (eval_count / self.budget)

            # Update velocities and positions using hybrid PSO-GA update rule
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (self.w * velocities +
                          adaptive_c1 * r1 * (personal_best_positions - positions) +
                          adaptive_c2 * r2 * (adaptive_memory - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Genetic Algorithm operations
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

            # Update global best with the latest evaluations
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

        return global_best_position