import numpy as np

class EnhancedQuantumLeapSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(30, dim * 6)  # Adjusted population for efficiency
        self.inertia_weight_init = 0.8
        self.inertia_weight_final = 0.3  # More dynamic final inertia for adaptability
        self.cognitive_coeff = 1.5  # Reduced to encourage exploration
        self.social_coeff = 2.0  # Increased social learning
        self.quantum_coeff = 0.7  # Amplified quantum behavior
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
            w = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * (1 - np.cos(phase * np.pi))  # Sinusoidal inertia decay
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)

            random_neighbor = personal_best_positions[np.random.randint(self.population_size)]
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (random_neighbor - particles))  # Random neighbor influence
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
                quantum_shift = np.random.uniform(-2.5, 2.5, (self.population_size, self.dim))  # Broader quantum shifts
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