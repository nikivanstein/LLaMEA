import numpy as np

class EnhancedQuantumCooperativeSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(40, dim * 6)
        self.inertia_weight_init = 0.9  # Increased for initial exploration
        self.inertia_weight_final = 0.3  # Reduced for final exploitation
        self.cognitive_coeff = 1.5  # Adjusted for balanced learning
        self.social_coeff = 2.0  # Enhanced for stronger social learning
        self.quantum_coeff = 0.5  # Adjusted for quantum frequency
        self.eval_count = 0
        self.chaotic_map = self._init_chaotic_map()

    def _init_chaotic_map(self):
        return np.random.rand(self.population_size)

    def _update_chaotic_map(self):
        self.chaotic_map = 4 * self.chaotic_map * (1 - self.chaotic_map)  # Logistic map

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
            self._update_chaotic_map()
            w = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * phase  # Linear decay
            adaptive_learning_rate = 0.5 + 0.5 * self.chaotic_map  # Adaptive learning rate using chaotic map
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * adaptive_learning_rate[:, None] * (personal_best_positions - particles) +
                          self.social_coeff * r2 * adaptive_learning_rate[:, None] * (global_best_position - particles))
            particles += velocities
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
                quantum_shift = np.random.uniform(-1.5, 1.5, (self.population_size, self.dim))  # Adjusted quantum leaps
                quantum_particles = particles + quantum_shift
                quantum_particles = np.clip(quantum_particles, self.lower_bound, self.upper_bound)

                for i in range(self.population_size):
                    if self.eval_count >= self.budget:
                        break
                    q_p = quantum_particles[i]
                    quantum_score = func(q_p)
                    self.eval_count += 1
                    if quantum_score < personal_best_scores[i]:
                        personal_best_positions[i] = q_p
                        personal_best_scores[i] = quantum_score
                        if quantum_score < global_best_score:
                            global_best_score = quantum_score
                            global_best_position = q_p

        return global_best_position