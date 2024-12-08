import numpy as np

class AQPSO_DLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.damping = 0.99
        self.quantum_prob = 0.1
        self.local_search_prob = 0.3
        self.adaptive_factor = 0.05  # Adaptive refinement factor

    def __call__(self, func):
        np.random.seed(0)

        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                self.inertia_weight *= self.damping

                if np.random.rand() < self.quantum_prob:
                    mean_best_position = np.mean(personal_best_positions, axis=0)
                    quantum_distance = np.abs(global_best_position - mean_best_position)
                    positions[i] = mean_best_position + quantum_distance * np.random.uniform(-0.5, 0.5, self.dim)
                else:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = (self.inertia_weight * velocities[i]
                                    + self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                                    + self.social_coeff * r2 * (global_best_position - positions[i]))
                    positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate new positions
                score = func(positions[i])
                evals += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Introduce Dual Local Search: global and personal
            if evals < self.budget:
                for _ in range(self.pop_size // 2):
                    if np.random.rand() < self.local_search_prob:
                        local_positions = global_best_position + np.random.normal(0, self.adaptive_factor, self.dim)
                    else:
                        local_positions = personal_best_positions[np.random.randint(self.pop_size)] + np.random.normal(0, self.adaptive_factor, self.dim)
                    local_positions = np.clip(local_positions, self.lower_bound, self.upper_bound)
                    local_score = func(local_positions)
                    evals += 1
                    if local_score < global_best_score:
                        global_best_score = local_score
                        global_best_position = local_positions

        return global_best_score