import numpy as np

class AdaptiveGuidedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = (0.9, 0.4)  # Adaptive inertia weight
        self.cognitive_weight = 1.4
        self.social_weight = 1.6
        self.strategy_switch = 0.5  # Probability of using local search

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            w = self.inertia_weight[1] + (self.inertia_weight[0] - self.inertia_weight[1]) * ((self.budget - evaluations) / self.budget)
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                          self.cognitive_weight * r1 * (personal_best_positions - positions) +
                          self.social_weight * r2 * (global_best_position - positions))
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

            if np.random.rand() < self.strategy_switch:
                for i in range(self.swarm_size):
                    if evaluations + 1 > self.budget:
                        break
                    local_search = positions[i] + np.random.uniform(-0.1, 0.1, self.dim)
                    local_search = np.clip(local_search, self.lb, self.ub)
                    local_search_score = func(local_search)
                    evaluations += 1

                    if local_search_score < personal_best_scores[i]:
                        personal_best_positions[i] = local_search
                        personal_best_scores[i] = local_search_score
                        if local_search_score < global_best_score:
                            global_best_position = local_search
                            global_best_score = local_search_score

        return global_best_position, global_best_score