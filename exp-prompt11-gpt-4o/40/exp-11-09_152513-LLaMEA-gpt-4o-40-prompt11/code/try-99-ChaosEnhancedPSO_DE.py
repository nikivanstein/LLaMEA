import numpy as np

class ChaosEnhancedPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40  # Reduced swarm for faster exploration
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.chaotic_map = self._logistic_map
        self.mutation_scale = 0.4  # Increased for stronger mutations
        self.F = 0.6  # Slightly larger scaling factor
        self.CR = 0.8  # Reduced crossover rate for diversity

    def _logistic_map(self, x):
        return 4 * x * (1 - x)

    def __call__(self, func):
        chaos_parameter = np.random.rand()
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size
        iter_count = 0

        while evaluations < self.budget:
            chaos_parameter = self.chaotic_map(chaos_parameter)
            r1, r2 = np.random.rand(2)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_weight * r1 * (personal_best_positions - positions) +
                          self.social_weight * r2 * (global_best_position - positions) +
                          chaos_parameter * np.random.uniform(-0.3, 0.3, velocities.shape))
            
            velocities = np.clip(velocities, -0.7, 0.7)
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)
            scores = np.array([func(p) for p in positions])
            evaluations += self.swarm_size

            better_idxs = scores < personal_best_scores
            personal_best_positions[better_idxs] = positions[better_idxs]
            personal_best_scores[better_idxs] = scores[better_idxs]

            current_global_best_idx = np.argmin(personal_best_scores)
            current_global_best_score = personal_best_scores[current_global_best_idx]

            if current_global_best_score < global_best_score:
                global_best_position = personal_best_positions[current_global_best_idx]
                global_best_score = current_global_best_score

            for i in range(self.swarm_size):
                indices = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant = a + self.F * (b - c)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, personal_best_positions[i])
                trial = np.clip(trial, self.lb, self.ub)
                trial_score = func(trial)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best_position = trial
                        global_best_score = trial_score

            iter_count += 1

        return global_best_position, global_best_score