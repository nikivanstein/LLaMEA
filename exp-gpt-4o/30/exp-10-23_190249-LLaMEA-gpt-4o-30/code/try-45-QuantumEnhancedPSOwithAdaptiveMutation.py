import numpy as np

class QuantumEnhancedPSOwithAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.inertia = 0.6  # Increased inertia for exploration
        self.cognitive = 1.5
        self.social = 2.2  # Increased social influence to drive convergence
        self.quantum_prob = 0.3  # Slightly higher probability for quantum leap
        self.global_best_position = None
        self.global_best_value = np.inf
        self.max_velocity = (self.ub - self.lb) * 0.15  # Reduced max velocity for better local exploration
        self.adaptive_factor = 0.8  # Introduced adaptive factor for mutation
        self.mutation_rate = 0.1  # Introduced lower mutation rate for stability

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

        while eval_count < self.budget:
            for i in range(self.num_particles):
                # Quantum-inspired update with adaptive enhancement
                if np.random.rand() < self.quantum_prob:
                    quantum_position = self.global_best_position + np.random.normal(0, self.adaptive_factor, self.dim)
                    quantum_position = np.clip(quantum_position, self.lb, self.ub)
                    quantum_value = func(quantum_position)
                    eval_count += 1
                    if quantum_value < personal_best_values[i]:
                        personal_best_values[i] = quantum_value
                        personal_best_positions[i] = quantum_position
                        
                # Velocity and position update
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.inertia * velocities[i] +
                    self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                    self.social * r2 * (self.global_best_position - positions[i])
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
                    
                # Adaptive Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = personal_best_positions[i] + np.random.normal(0, self.adaptive_factor, self.dim)
                    mutation_vector = np.clip(mutation_vector, self.lb, self.ub)
                    mutation_value = func(mutation_vector)
                    eval_count += 1
                    if mutation_value < personal_best_values[i]:
                        personal_best_positions[i] = mutation_vector
                        personal_best_values[i] = mutation_value

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