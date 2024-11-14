import numpy as np

class EnhancedPSO_AM_DM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 50  # Increased for broader exploration
        self.w = 0.5  # Moderately increased inertia for better balance
        self.c1 = 1.8  # Reduced cognitive component to prevent over-exploration
        self.c2 = 2.0  # Adjusted social component for stronger global pull
        self.F = 0.9  # Increased mutation factor for enhanced exploration
        self.CR = 0.85  # Slightly reduced crossover rate for stability
        self.adaptive_lr = 100  # Lowered adaptive learning rate scale for quick adaptability

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.num_particles, self.dim))  # More controlled initial velocity
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
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Dynamic adaptive learning rate
            adaptive_w = self.w / (1 + (eval_count / self.adaptive_lr) ** 2)  # Faster decay for quicker adaptation
            velocities *= adaptive_w

            # Apply Adaptive Differential Mutation with memory
            memory = np.copy(positions)
            for i in range(self.num_particles):
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = memory[a] + self.F * (memory[b] - memory[c])
                    mutant = np.clip(mutant, self.lb, self.ub)
                    
                    if func(mutant) < scores[i]:
                        positions[i] = mutant
                        eval_count += 1
                        if eval_count >= self.budget:
                            break

        return global_best_position