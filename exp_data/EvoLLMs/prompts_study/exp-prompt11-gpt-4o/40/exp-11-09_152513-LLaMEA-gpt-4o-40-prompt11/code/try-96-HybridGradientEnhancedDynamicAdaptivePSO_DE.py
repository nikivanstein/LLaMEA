import numpy as np

class HybridGradientEnhancedDynamicAdaptivePSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 45  # Adjusted swarm size
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight_initial = 0.9  # Slightly increased to enhance exploration
        self.inertia_weight_final = 0.4
        self.cognitive_weight_initial = 1.2
        self.cognitive_weight_final = 1.0
        self.social_weight_initial = 1.5
        self.social_weight_final = 2.0
        self.mutation_scale = 0.3  # Further increased for diversity
        self.F = 0.6  # Increased DE scaling factor
        self.CR = 0.85  # Slightly reduced DE crossover rate
        self.gradient_step = 0.1  # Step for gradient-based local search

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
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

            # Differential Evolution-like perturbation and gradient-based local search
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

                # Gradient-based local search
                gradient = np.zeros(self.dim)
                for d in range(self.dim):
                    perturbed = np.copy(personal_best_positions[i])
                    perturbed[d] += self.gradient_step
                    gradient[d] = (func(perturbed) - personal_best_scores[i]) / self.gradient_step
                num_gradients = 1
                improved_position = personal_best_positions[i] - self.gradient_step * gradient
                improved_position = np.clip(improved_position, self.lb, self.ub)
                improved_score = func(improved_position)
                evaluations += num_gradients

                if improved_score < personal_best_scores[i]:
                    personal_best_positions[i] = improved_position
                    personal_best_scores[i] = improved_score
                    if improved_score < global_best_score:
                        global_best_position = improved_position
                        global_best_score = improved_score

            iter_count += 1

        return global_best_position, global_best_score