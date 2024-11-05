import numpy as np

class QuantumEnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(50, dim * 5)  # Adjusted population for better exploration
        self.inertia_weight_init = 0.8
        self.inertia_weight_final = 0.5  # Adjusted final inertia for dynamic adaptation
        self.cognitive_coeff = 2.0  # Refined balance for cognitive learning
        self.social_coeff = 2.0  # Enhanced social coefficient for global learning
        self.quantum_coeff = 0.7  # Refined quantum behavior for diversification
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
            w = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * np.sin(phase * np.pi / 2)  # Sine inertia decay for smoother transition
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
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

            # Dual-layer quantum adaptation for enhanced exploration
            if np.random.rand() < self.quantum_coeff:
                quantum_shift1 = np.random.uniform(-1, 1, (self.population_size, self.dim))
                quantum_shift2 = np.random.uniform(-3, 3, (self.population_size, self.dim))
                quantum_particles1 = particles + quantum_shift1
                quantum_particles2 = particles + quantum_shift2
                quantum_particles1 = np.clip(quantum_particles1, self.lower_bound, self.upper_bound)
                quantum_particles2 = np.clip(quantum_particles2, self.lower_bound, self.upper_bound)
                
                for i in range(self.population_size):
                    if self.eval_count >= self.budget:
                        break
                    q_p1, q_p2 = quantum_particles1[i], quantum_particles2[i]
                    quantum_score1 = func(q_p1)
                    quantum_score2 = func(q_p2)
                    self.eval_count += 2
                    if quantum_score1 < personal_best_scores[i]:
                        personal_best_positions[i] = q_p1
                        personal_best_scores[i] = quantum_score1
                        if quantum_score1 < global_best_score:
                            global_best_score = quantum_score1
                            global_best_position = q_p1
                    if quantum_score2 < personal_best_scores[i]:
                        personal_best_positions[i] = q_p2
                        personal_best_scores[i] = quantum_score2
                        if quantum_score2 < global_best_score:
                            global_best_score = quantum_score2
                            global_best_position = q_p2

        return global_best_position