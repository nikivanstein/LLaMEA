import numpy as np

class EnhancedHybridPSO_ADM_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 35  # Slightly increased number of particles
        self.initial_w = 0.7  # Higher initial inertia weight
        self.final_w = 0.3  # Lower final inertia weight
        self.c1 = 1.5  # Adjusted cognitive component
        self.c2 = 2.5  # Increased social component
        self.F = 0.8  # Slightly reduced differential mutation factor
        self.CR = 0.9  # Increased crossover probability
        self.adaptive_lr = 100  # Retained adaptive learning rate scale factor

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
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

            # Compute dynamic inertia weight
            w = self.initial_w - (self.initial_w - self.final_w) * (eval_count / self.budget)

            # Update velocities and positions using PSO update rule
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Adaptive Differential Mutation with dynamic F
            for i in range(self.num_particles):
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    dynamic_F = self.F * (1 - eval_count / self.budget)
                    mutant = positions[a] + dynamic_F * (positions[b] - positions[c])
                    mutant = np.clip(mutant, self.lb, self.ub)

                    mutant_score = func(mutant)
                    eval_count += 1

                    if mutant_score < scores[i]:
                        positions[i] = mutant
                        scores[i] = mutant_score

                    if eval_count >= self.budget:
                        break

        return global_best_position