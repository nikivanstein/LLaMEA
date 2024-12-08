import numpy as np

class AdaptiveNonlinearPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50  # Increased swarm size for better exploration
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.9  # Start with high inertia for exploration
        self.inertia_damping = 0.99  # Nonlinear inertia reduction
        self.cognitive_weight = 1.5  # Adjusted cognition and social weights
        self.social_weight = 1.5
        self.local_search_prob = 0.4  # Increased probability for local search

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2)
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
                    mutant = x1 + 0.85 * (x2 - x3)  # Adjusted mutation factor
                    mutant = np.clip(mutant, self.lb, self.ub)
                    
                    if np.random.rand() < self.local_search_prob:  # Enhanced local search
                        local_search = mutant + np.random.normal(0, 0.2, self.dim)  # Increased search radius
                        local_search = np.clip(local_search, self.lb, self.ub)
                        mutant = local_search

                    mutant_score = func(mutant)
                    evaluations += 1

                    if mutant_score < personal_best_scores[i]:
                        personal_best_positions[i] = mutant
                        personal_best_scores[i] = mutant_score

                        if mutant_score < global_best_score:
                            global_best_position = mutant
                            global_best_score = mutant_score

            # Nonlinear inertia reduction
            self.inertia_weight *= self.inertia_damping

        return global_best_position, global_best_score