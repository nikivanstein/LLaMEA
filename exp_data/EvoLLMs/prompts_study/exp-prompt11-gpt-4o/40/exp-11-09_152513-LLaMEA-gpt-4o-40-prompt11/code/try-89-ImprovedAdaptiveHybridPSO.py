import numpy as np

class ImprovedAdaptiveHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 70  # Increased swarm size for a better search space sampling
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight_initial = 0.8
        self.inertia_weight_final = 0.3
        self.cognitive_weight_initial = 2.2  # Enhanced cognitive component for balance
        self.cognitive_weight_final = 1.3
        self.social_weight_initial = 1.3
        self.social_weight_final = 2.3
        self.mutation_scale = 0.25  # Larger mutation scale for diverse local search

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.4, 0.4, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size
        dynamic_inertia = lambda iter: self.inertia_weight_initial - iter * (self.inertia_weight_initial - self.inertia_weight_final) / (self.budget/self.swarm_size)
        cognitive_weight = lambda iter: self.cognitive_weight_initial - iter * (self.cognitive_weight_initial - self.cognitive_weight_final) / (self.budget/self.swarm_size)
        social_weight = lambda iter: self.social_weight_initial + iter * (self.social_weight_final - self.social_weight_initial) / (self.budget/self.swarm_size)

        phase_threshold = self.budget // 3  # Adjusted phase threshold for better adaptation

        iter_count = 0

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
            velocities = (dynamic_inertia(iter_count) * velocities +
                          cognitive_weight(iter_count) * r1 * (personal_best_positions - positions) +
                          social_weight(iter_count) * r2 * (global_best_position - positions))
            
            if evaluations < phase_threshold:
                velocities *= 0.75  # More exploration in initial stage
            elif evaluations >= 2 * phase_threshold:
                velocities *= 1.25  # Stronger convergence in final stage
            
            velocities = np.clip(velocities, -0.5, 0.5)  # Tighter velocity limits for improved control
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

            if evaluations + self.swarm_size <= self.budget:
                elite_idxs = np.argsort(personal_best_scores)[:15]  # Broader elite sample for local enhancement
                for i in elite_idxs:
                    mutant1 = personal_best_positions[i] + self.mutation_scale * np.random.randn(self.dim)
                    mutant2 = global_best_position + self.mutation_scale * np.random.randn(self.dim)  # Additional mutation strategy
                    mutant1 = np.clip(mutant1, self.lb, self.ub)
                    mutant2 = np.clip(mutant2, self.lb, self.ub)
                    mutant_score1 = func(mutant1)
                    mutant_score2 = func(mutant2)
                    evaluations += 2

                    if mutant_score1 < personal_best_scores[i] or mutant_score2 < personal_best_scores[i]:
                        if mutant_score1 < mutant_score2:
                            personal_best_positions[i] = mutant1
                            personal_best_scores[i] = mutant_score1
                        else:
                            personal_best_positions[i] = mutant2
                            personal_best_scores[i] = mutant_score2

                        if min(mutant_score1, mutant_score2) < global_best_score:
                            global_best_position = mutant1 if mutant_score1 < mutant_score2 else mutant2
                            global_best_score = min(mutant_score1, mutant_score2)

            iter_count += 1

        return global_best_position, global_best_score