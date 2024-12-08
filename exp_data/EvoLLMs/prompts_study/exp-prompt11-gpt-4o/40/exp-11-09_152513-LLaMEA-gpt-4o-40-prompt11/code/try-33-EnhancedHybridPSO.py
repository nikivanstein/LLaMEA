import numpy as np

class EnhancedHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.9
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.min_inertia_weight = 0.4
        self.max_inertia_weight = 0.9

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
            r1, r2 = np.random.rand(2)
            inertia_decay = (self.max_inertia_weight - self.min_inertia_weight) * (1 - evaluations / self.budget)
            self.inertia_weight = self.min_inertia_weight + inertia_decay

            velocities = (self.inertia_weight * velocities +
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

            if evaluations + self.swarm_size * 3 <= self.budget:
                for i in range(self.swarm_size):
                    idxs = np.random.choice(np.arange(self.swarm_size), 3, replace=False)
                    x1, x2, x3 = positions[idxs]
                    mutant = x1 + 0.8 * (x2 - x3)
                    mutant = np.clip(mutant, self.lb, self.ub)

                    chaotic_factor = np.random.uniform(0, 1)
                    chaotic_mutant = mutant + chaotic_factor * (global_best_position - mutant)
                    chaotic_mutant = np.clip(chaotic_mutant, self.lb, self.ub)

                    mutant_score = func(mutant)
                    chaotic_score = func(chaotic_mutant)
                    evaluations += 2

                    if mutant_score < personal_best_scores[i]:
                        personal_best_positions[i] = mutant
                        personal_best_scores[i] = mutant_score

                        if mutant_score < global_best_score:
                            global_best_position = mutant
                            global_best_score = mutant_score

                    if chaotic_score < personal_best_scores[i]:
                        personal_best_positions[i] = chaotic_mutant
                        personal_best_scores[i] = chaotic_score

                        if chaotic_score < global_best_score:
                            global_best_position = chaotic_mutant
                            global_best_score = chaotic_score

        return global_best_position, global_best_score