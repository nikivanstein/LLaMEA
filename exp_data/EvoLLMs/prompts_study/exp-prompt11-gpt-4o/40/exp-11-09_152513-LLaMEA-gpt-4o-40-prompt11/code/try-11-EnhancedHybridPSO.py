import numpy as np

class EnhancedHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.lb = -5.0
        self.ub = 5.0
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0
        self.inertia_min = 0.4
        self.inertia_max = 0.9

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
            # Dynamically adjust inertia weight
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))

            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = (self.inertia_weight * velocities +
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

            # Quantum-inspired Mutation
            if evaluations + self.swarm_size <= self.budget:
                for i in range(self.swarm_size):
                    quantum_step = np.random.uniform(self.lb, self.ub, self.dim)
                    quantum_position = global_best_position + 0.5 * (quantum_step - positions[i])
                    quantum_position = np.clip(quantum_position, self.lb, self.ub)

                    quantum_score = func(quantum_position)
                    evaluations += 1

                    if quantum_score < personal_best_scores[i]:
                        personal_best_positions[i] = quantum_position
                        personal_best_scores[i] = quantum_score

                        if quantum_score < global_best_score:
                            global_best_position = quantum_position
                            global_best_score = quantum_score

        return global_best_position, global_best_score