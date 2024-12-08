import numpy as np

class EnhancedHybridPSO_ADM_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 35  # Slightly increased number of particles for better exploration
        self.w = 0.4  # Further reduced inertia weight for faster convergence
        self.c1_initial = 1.5  # Dynamic cognitive component
        self.c2_initial = 2.5  # Dynamic social component
        self.F = 0.8  # Adjusted differential mutation factor
        self.CR = 0.9  # Increased crossover probability for diversity
        self.adaptive_lr = 150  # Modified adaptive learning rate scale factor
        self.alpha = 0.99  # Decay factor for dynamic parameters

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

            # Update velocities and positions using PSO update rule
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            c1 = self.c1_initial * (self.alpha ** (eval_count / self.budget))
            c2 = self.c2_initial * (self.alpha ** (eval_count / self.budget))
            velocities = (self.w * velocities +
                          c1 * r1 * (personal_best_positions - positions) +
                          c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Adaptive learning rate
            adaptive_w = self.w / (1 + eval_count / self.adaptive_lr)
            velocities *= adaptive_w

            # Improved Adaptive Differential Mutation
            for i in range(self.num_particles):
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = positions[a] + self.F * (personal_best_positions[b] - personal_best_positions[c])
                    mutant = np.clip(mutant, self.lb, self.ub)

                    mutant_score = func(mutant)
                    if mutant_score < scores[i]:
                        positions[i] = mutant
                        eval_count += 1
                        if eval_count >= self.budget:
                            break

        return global_best_position