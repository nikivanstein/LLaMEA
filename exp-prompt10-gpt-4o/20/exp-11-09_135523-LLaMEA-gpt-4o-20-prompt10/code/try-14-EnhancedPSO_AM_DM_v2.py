import numpy as np

class EnhancedPSO_AM_DM_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 60  # Increased for wider exploration
        self.w = 0.4  # Adjusted inertia for more responsiveness
        self.c1 = 1.5  # Further reduced cognitive component
        self.c2 = 2.5  # Enhanced social component for stronger attraction to global best
        self.F = 0.8  # Modified mutation factor for balanced exploration
        self.CR = 0.9  # Increased crossover rate for more frequent mutations
        self.adaptive_lr = 80  # Further lowered adaptive learning rate scale

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.2, 0.2, (self.num_particles, self.dim))  # Tighter initial velocity range
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
            adaptive_w = self.w / (1 + (eval_count / self.adaptive_lr) ** 2)  # Similar decay for adaptability
            velocities *= adaptive_w

            # Apply Adaptive Differential Mutation with memory
            memory = np.copy(personal_best_positions)  # Use personal bests for memory
            for i in range(self.num_particles):
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = memory[a] + self.F * (memory[b] - memory[c])
                    mutant = np.clip(mutant, self.lb, self.ub)
                    
                    mutant_score = func(mutant)
                    eval_count += 1
                    if mutant_score < scores[i]:
                        positions[i] = mutant
                        scores[i] = mutant_score

                    if eval_count >= self.budget:
                        break

            # Update global best with the latest evaluations
            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

        return global_best_position