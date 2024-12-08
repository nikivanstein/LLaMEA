import numpy as np

class HybridPSO_AQIM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 60
        self.w = 0.5  # Increased inertia for maintaining exploration early on
        self.c1 = 1.2  # Slightly reduced cognitive component for less oscillation
        self.c2 = 2.8  # Further enhanced social component for stronger convergence
        self.F = 0.9  # Increased mutation factor for more aggressive diversity
        self.CR = 0.85  # Reduced crossover rate to balance mutation application
        self.adaptive_lr = 100  # Slightly higher adaptive learning scale for gradual decay

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.num_particles, self.dim))  # Wider initial velocity range
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

            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)

            adaptive_w = self.w / (1 + (eval_count / self.adaptive_lr))
            velocities *= adaptive_w

            memory = np.copy(personal_best_positions)
            for i in range(self.num_particles):
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = memory[a] + self.F * (memory[b] - memory[c])
                    mutant = np.clip(mutant, self.lb, self.ub)
                    
                    quantum_shift = np.random.normal(0, 0.1, self.dim)  # Quantum-inspired shift
                    quantum_mutant = mutant + quantum_shift
                    quantum_mutant = np.clip(quantum_mutant, self.lb, self.ub)

                    mutant_score = func(quantum_mutant)
                    eval_count += 1
                    if mutant_score < scores[i]:
                        positions[i] = quantum_mutant
                        scores[i] = mutant_score

                    if eval_count >= self.budget:
                        break

            current_best_idx = np.argmin(scores)
            if scores[current_best_idx] < global_best_score:
                global_best_score = scores[current_best_idx]
                global_best_position = positions[current_best_idx]

        return global_best_position