import numpy as np

class DualStrategyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.9  # Increased inertia for exploration
        self.cognitive_weight = 1.4
        self.social_weight = 1.6
        self.chaotic_factor = 0.5  # Chaotic factor for dynamic modification

    def __call__(self, func):
        # Initialize particles
        positions = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.swarm_size
        chaotic_sequence = np.random.rand(self.swarm_size)

        while evaluations < self.budget:
            # Update velocities and positions with chaotic map
            r1, r2 = np.random.rand(2)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_weight * r1 * (personal_best_positions - positions) +
                          self.social_weight * r2 * (global_best_position - positions) +
                          self.chaotic_factor * chaotic_sequence.reshape(-1, 1) * (np.random.rand(self.swarm_size, self.dim) - 0.5))
            positions += velocities

            # Bound positions
            positions = np.clip(positions, self.lb, self.ub)

            # Evaluate new positions
            scores = np.array([func(p) for p in positions])
            evaluations += self.swarm_size

            # Update personal bests
            better_idxs = scores < personal_best_scores
            personal_best_positions[better_idxs] = positions[better_idxs]
            personal_best_scores[better_idxs] = scores[better_idxs]

            # Update global best
            current_global_best_idx = np.argmin(personal_best_scores)
            current_global_best_score = personal_best_scores[current_global_best_idx]

            if current_global_best_score < global_best_score:
                global_best_position = personal_best_positions[current_global_best_idx]
                global_best_score = current_global_best_score

            # Crowding mechanism for diversity
            for i in range(self.swarm_size):
                candidate = positions[i] + np.random.uniform(-0.1, 0.1, self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_score = func(candidate)
                evaluations += 1

                if candidate_score < personal_best_scores[i]:
                    personal_best_positions[i] = candidate
                    personal_best_scores[i] = candidate_score

                    if candidate_score < global_best_score:
                        global_best_position = candidate
                        global_best_score = candidate_score

        return global_best_position, global_best_score