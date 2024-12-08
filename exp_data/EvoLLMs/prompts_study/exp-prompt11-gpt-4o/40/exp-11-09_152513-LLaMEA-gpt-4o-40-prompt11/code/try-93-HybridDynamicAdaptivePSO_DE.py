import numpy as np

class HybridDynamicAdaptivePSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50  # Reduced for quicker convergence
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight_initial = 0.8
        self.inertia_weight_final = 0.3
        self.cognitive_weight_initial = 1.4  # Further adaptive cognitive component
        self.cognitive_weight_final = 1.2
        self.social_weight_initial = 1.2  # Further adaptive social component
        self.social_weight_final = 1.8
        self.mutation_scale = 0.25  # Adjusted for more impactful mutations
        self.F = 0.5  # Differential evolution scaling factor
        self.CR = 0.9  # Differential evolution crossover rate

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size
        dynamic_inertia = lambda iter: self.inertia_weight_initial - iter * (self.inertia_weight_initial - self.inertia_weight_final) / (self.budget/self.swarm_size)
        cognitive_weight = lambda iter: self.cognitive_weight_initial - iter * (self.cognitive_weight_initial - self.cognitive_weight_final) / (self.budget/self.swarm_size)
        social_weight = lambda iter: self.social_weight_initial + iter * (self.social_weight_final - self.social_weight_initial) / (self.budget/self.swarm_size)
        iter_count = 0

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
            velocities = (dynamic_inertia(iter_count) * velocities +
                          cognitive_weight(iter_count) * r1 * (personal_best_positions - positions) +
                          social_weight(iter_count) * r2 * (global_best_position - positions))
            
            velocities = np.clip(velocities, -0.7, 0.7)  # Adjusted velocity limits
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

            # Differential Evolution-like perturbation for additional diversity
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