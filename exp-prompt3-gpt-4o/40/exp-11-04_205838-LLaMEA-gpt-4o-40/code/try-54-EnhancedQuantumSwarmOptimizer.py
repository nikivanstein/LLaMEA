import numpy as np

class EnhancedQuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(40, dim * 7)  # Slightly increased population for enhanced diversity
        self.inertia_weight_init = 0.9  # Higher initial inertia weight for global exploration
        self.inertia_weight_final = 0.4  # Increased final inertia weight for maintaining some exploration
        self.cognitive_coeff = 1.5  # Reduced cognitive coefficient for less local focus
        self.social_coeff = 1.7  # Increased social coefficient for improved convergence
        self.quantum_coeff = 0.3  # Reduced quantum coefficient for balance
        self.mutation_rate = 0.1  # Added mutation for diversity
        self.eval_count = 0

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        self.eval_count += self.population_size

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        while self.eval_count < self.budget:
            phase = self.eval_count / self.budget
            w = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * phase
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Mutation step to enhance exploration
            mutation = np.random.uniform(-0.5, 0.5, particles.shape) < self.mutation_rate
            particles += mutation * np.random.uniform(-1, 1, particles.shape)
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            scores = np.array([func(p) for p in particles])
            self.eval_count += self.population_size

            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = particles[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_index] < global_best_score:
                global_best_score = personal_best_scores[global_best_index]
                global_best_position = personal_best_positions[global_best_index]

            if np.random.rand() < self.quantum_coeff:
                quantum_shift = np.random.uniform(-1, 1, (self.population_size, self.dim))
                quantum_particles = particles + quantum_shift
                quantum_particles = np.clip(quantum_particles, self.lower_bound, self.upper_bound)

                for i, q_p in enumerate(quantum_particles):
                    if self.eval_count >= self.budget:
                        break
                    quantum_score = func(q_p)
                    self.eval_count += 1
                    if quantum_score < personal_best_scores[i]:
                        personal_best_positions[i] = q_p
                        personal_best_scores[i] = quantum_score
                        if quantum_score < global_best_score:
                            global_best_score = quantum_score
                            global_best_position = q_p
        
        return global_best_position