import numpy as np

class HybridQuantumSelfOrganizingDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.inertia = 0.4  # Slightly reduced inertia for more exploration
        self.cognitive = 1.5
        self.social = 1.8
        self.quantum_prob = 0.3  # Increased quantum probability
        self.global_best_position = None
        self.global_best_value = np.inf
        self.max_velocity = (self.ub - self.lb) * 0.15  # Reduced max velocity
        self.de_mutation_factor = 0.9
        self.de_crossover_rate = 0.85  # Adjusted crossover rate for diversity

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.inf)
        som_learning_rate = 0.3  # Self-organizing map learning rate

        for i in range(self.num_particles):
            value = func(positions[i])
            personal_best_values[i] = value
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = np.copy(positions[i])

        eval_count = self.num_particles

        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Quantum-inspired update with self-organizing map influence
                if np.random.rand() < self.quantum_prob:
                    quantum_position = self.global_best_position + np.random.randn(self.dim)
                    quantum_position = np.clip(quantum_position, self.lb, self.ub)
                    quantum_value = func(quantum_position)
                    eval_count += 1
                    if quantum_value < personal_best_values[i]:
                        personal_best_values[i] = quantum_value
                        personal_best_positions[i] = quantum_position
                
                # Velocity and position update with SOM effect
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.inertia * velocities[i] +
                    self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                    self.social * r2 * (self.global_best_position - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                # Self-organizing map adaptation
                som_adjustment = som_learning_rate * (self.global_best_position - positions[i])
                positions[i] += som_adjustment

                # Evaluate new position
                value = func(positions[i])
                eval_count += 1
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = np.copy(positions[i])
                    
                # Differential Mutation with SOM influence
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