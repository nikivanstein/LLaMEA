import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.5  # Changed for chaotic dynamics
        self.cognitive_weight = 1.8  # Increased cognitive influence
        self.social_weight = 1.8  # Increased social influence
        self.chaos_sequence = np.random.rand(self.budget)

    def chaotic_inertia(self, step):
        return 0.5 + 0.5 * np.sin(2.0 * np.pi * self.chaos_sequence[step % len(self.chaos_sequence)])

    def opposition_based_learning(self, positions):
        return self.lb + self.ub - positions

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

        while evaluations < self.budget:
            inertia_weight = self.chaotic_inertia(evaluations)
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = (inertia_weight * velocities +
                          self.cognitive_weight * r1 * (personal_best_positions - positions) +
                          self.social_weight * r2 * (global_best_position - positions))
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

            # Apply opposition-based learning
            opposition_positions = self.opposition_based_learning(positions)
            opposition_scores = np.array([func(p) for p in opposition_positions])
            evaluations += self.swarm_size

            opposition_better_idxs = opposition_scores < personal_best_scores
            personal_best_positions[opposition_better_idxs] = opposition_positions[opposition_better_idxs]
            personal_best_scores[opposition_better_idxs] = opposition_scores[opposition_better_idxs]

            if evaluations + self.swarm_size * 3 <= self.budget:
                for i in range(self.swarm_size):
                    idxs = np.random.choice(np.arange(self.swarm_size), 3, replace=False)
                    x1, x2, x3 = positions[idxs]
                    mutant = x1 + 0.8 * (x2 - x3)
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