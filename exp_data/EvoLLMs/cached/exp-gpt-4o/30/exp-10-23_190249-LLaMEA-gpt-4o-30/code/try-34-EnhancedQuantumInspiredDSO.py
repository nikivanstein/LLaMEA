import numpy as np

class EnhancedQuantumInspiredDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.inertia = 0.5
        self.cognitive = 1.5
        self.social = 1.5
        self.quantum_prob = 0.3  # Increased initial quantum probability
        self.global_best_position = None
        self.global_best_value = np.inf
        self.max_velocity = (self.ub - self.lb) * 0.2
        self.de_mutation_factor = 0.9  # Adjusted DE mutation factor
        self.de_crossover_rate = 0.85  # Adjusted DE crossover rate

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.inf)

        for i in range(self.num_particles):
            value = func(positions[i])
            personal_best_values[i] = value
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = np.copy(positions[i])

        eval_count = self.num_particles
        dynamic_quantum_prob = self.quantum_prob

        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Dynamic Quantum-inspired update
                if np.random.rand() < dynamic_quantum_prob:
                    quantum_position = self.global_best_position + np.random.randn(self.dim)
                    quantum_position = np.clip(quantum_position, self.lb, self.ub)
                    quantum_value = func(quantum_position)
                    eval_count += 1
                    if quantum_value < personal_best_values[i]:
                        personal_best_values[i] = quantum_value
                        personal_best_positions[i] = quantum_position
                        # Reduce quantum probability as particles improve
                        dynamic_quantum_prob = max(0.1, dynamic_quantum_prob * 0.95)
                
                # Velocity and position update with adaptive learning rates
                r1, r2 = np.random.rand(), np.random.rand()
                adaptive_cognitive = self.cognitive * (1 + np.exp(-0.1 * eval_count / self.budget))
                adaptive_social = self.social * (1 + np.exp(-0.1 * eval_count / self.budget))
                velocities[i] = (
                    self.inertia * velocities[i] +
                    adaptive_cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                    adaptive_social * r2 * (self.global_best_position - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                # Evaluate new position
                value = func(positions[i])
                eval_count += 1
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = np.copy(positions[i])
                    
                # Differential Mutation with adaptive factor
                if np.random.rand() < self.de_crossover_rate:
                    idxs = np.random.choice(self.num_particles, 3, replace=False)
                    donor_vector = (
                        positions[idxs[0]] +
                        self.de_mutation_factor * (positions[idxs[1]] - positions[idxs[2]])
                    )
                    trial_vector = np.where(np.random.rand(self.dim) < self.de_crossover_rate,
                                            donor_vector, positions[i])
                    trial_vector = np.clip(trial_vector, self.lb, self.ub)
                    
                    trial_value = func(trial_vector)
                    eval_count += 1
                    if trial_value < personal_best_values[i]:
                        positions[i] = trial_vector
                        personal_best_positions[i] = trial_vector
                        personal_best_values[i] = trial_value

                if eval_count >= self.budget:
                    break

            if eval_count >= self.budget:
                break

            # Update global best
            for i in range(self.num_particles):
                value = personal_best_values[i]
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(personal_best_positions[i])

        return self.global_best_position, self.global_best_value