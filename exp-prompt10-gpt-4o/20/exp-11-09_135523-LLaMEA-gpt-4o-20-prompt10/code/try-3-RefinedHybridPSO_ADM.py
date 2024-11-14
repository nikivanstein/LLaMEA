import numpy as np

class RefinedHybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 35  # Slightly increased number of particles for diversity
        self.w = 0.4  # Further reduced inertia weight for faster convergence
        self.c1 = 2.2  # Enhanced cognitive component for increased individual search
        self.c2 = 1.8  # Slightly reduced social component for less global dependence
        self.F_min = 0.5  # Lower bound for differential mutation scale
        self.F_max = 0.9  # Upper bound for differential mutation scale
        self.CR = 0.9  # Increased crossover probability for more exploration
        self.adaptive_lr = 150  # Adjusted adaptive learning rate scale factor

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

            # Calculate diversity-based inertia weight
            diversity = np.mean(np.std(positions, axis=0))
            w_dynamic = self.w * (1 + diversity / self.dim)

            # Update velocities and positions using PSO update rule
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (w_dynamic * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            # Adaptive differential mutation scale
            F_dynamic = self.F_max - (self.F_max - self.F_min) * (eval_count / self.budget)

            # Apply Adaptive Differential Mutation
            for i in range(self.num_particles):
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = positions[a] + F_dynamic * (positions[b] - positions[c])
                    mutant = np.clip(mutant, self.lb, self.ub)
                    
                    if func(mutant) < scores[i]:
                        positions[i] = mutant
                        eval_count += 1
                        if eval_count >= self.budget:
                            break

        return global_best_position