import numpy as np

class EnhancedDynamicAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 60  # Increased for diversified exploration
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_weight_initial = 2.0  # Adaptive cognitive component
        self.cognitive_weight_final = 1.5
        self.social_weight_initial = 1.5  # Adaptive social component
        self.social_weight_final = 2.0
        self.mutation_scale = 0.15  # Adjusted mutation scale for finer search

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
        phase_threshold = self.budget // 4  # Modified phase threshold

        iter_count = 0

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
            velocities = (dynamic_inertia(iter_count) * velocities +
                          cognitive_weight(iter_count) * r1 * (personal_best_positions - positions) +
                          social_weight(iter_count) * r2 * (global_best_position - positions))
            
            if evaluations < phase_threshold:
                velocities *= 0.85  # Slightly more exploration initially
            elif evaluations >= 3 * phase_threshold:
                velocities *= 1.15  # Aggressive convergence in the final phase
            
            velocities = np.clip(velocities, -0.6, 0.6)  # Adjusted velocity limits for stability
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

            if evaluations + self.swarm_size * 2 <= self.budget:
                elite_idxs = np.argsort(personal_best_scores)[:10]  # Increased elite count for broader local search
                for i in elite_idxs:
                    mutant = personal_best_positions[i] + self.mutation_scale * np.random.randn(self.dim)
                    mutant = np.clip(mutant, self.lb, self.ub)
                    mutant_score = func(mutant)
                    evaluations += 1

                    if mutant_score < personal_best_scores[i]:
                        personal_best_positions[i] = mutant
                        personal_best_scores[i] = mutant_score

                        if mutant_score < global_best_score:
                            global_best_position = mutant
                            global_best_score = mutant_score

            iter_count += 1

        return global_best_position, global_best_score