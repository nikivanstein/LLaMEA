import numpy as np

class EnhancedHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_weight = 1.5
        self.social_weight = 1.5

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size
        stagnation_counter = 0
        max_stagnation = 10

        while evaluations < self.budget:
            inertia_weight = self.inertia_weight_initial - (
                (self.inertia_weight_initial - self.inertia_weight_final) * (evaluations / self.budget))

            r1, r2 = np.random.rand(2)
            velocities = (inertia_weight * velocities +
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
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= max_stagnation and evaluations + self.swarm_size * 2 <= self.budget:
                for i in range(self.swarm_size):
                    idxs = np.random.choice(np.arange(self.swarm_size), 3, replace=False)
                    x1, x2, x3 = positions[idxs]
                    if np.random.rand() < 0.5:
                        mutant = x1 + 0.5 * (x2 - x3)
                    else:
                        neighbor_idxs = np.random.choice(np.delete(np.arange(self.swarm_size), i), 3, replace=False)
                        n1, n2, n3 = positions[neighbor_idxs]
                        mutant = n1 + 0.5 * (n2 - n3)
                    
                    mutant = np.clip(mutant, self.lb, self.ub)
                    mutant_score = func(mutant)
                    evaluations += 1

                    if mutant_score < personal_best_scores[i]:
                        personal_best_positions[i] = mutant
                        personal_best_scores[i] = mutant_score

                        if mutant_score < global_best_score:
                            global_best_position = mutant
                            global_best_score = mutant_score

        return global_best_position, global_best_score