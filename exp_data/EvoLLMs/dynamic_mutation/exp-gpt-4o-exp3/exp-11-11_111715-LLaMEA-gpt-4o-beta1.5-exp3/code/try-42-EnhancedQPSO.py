import numpy as np

class EnhancedQPSO:
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
        
        # Dynamic quantum probability starting lower
        self.quantum_prob = 0.05  
        self.quantum_prob_step = 0.05 / (0.5 * self.budget)  # Increase step
        
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
                # Adjust inertia and coefficients adaptively
                self.inertia_weight *= self.damping

                # Dynamic Quantum-inspired Update
                if np.random.rand() < self.quantum_prob:
                    mean_best_position = np.mean(personal_best_positions, axis=0)
                    quantum_distance = np.abs(global_best_position - mean_best_position)
                    positions[i] = mean_best_position + quantum_distance * np.random.uniform(-0.5, 0.5, self.dim)
                else:
                    # Update velocities and positions
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

            # Increase quantum probability gradually to enhance exploration
            self.quantum_prob = min(0.5, self.quantum_prob + self.quantum_prob_step)

            # Differential perturbation for diversity
            if evals < self.budget:
                diff_perturbation_factor = 0.8
                for j in range(self.pop_size // 2):
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    a, b, c = personal_best_positions[indices]
                    diff_vector = diff_perturbation_factor * (b - c)
                    trial_position = np.clip(a + diff_vector, self.lower_bound, self.upper_bound)
                    trial_score = func(trial_position)
                    evals += 1
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_position

        return global_best_score