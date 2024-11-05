import numpy as np

class QuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(50, dim * 7)  # Increased population for diversity
        self.inertia_weight_init = 0.8  # Slightly increased initial inertia for exploration
        self.inertia_weight_final = 0.3  # Reduced final inertia for faster convergence
        self.cognitive_coeff = 2.0  # Enhanced cognitive component
        self.social_coeff = 1.5  # Reduced social coefficient for individual exploration
        self.quantum_coeff = 0.7  # Increased quantum behavior for diverse search
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
            w = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * np.sin(phase * np.pi / 2)  # Sine inertia decay
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

            local_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[local_best_index] < global_best_score:
                global_best_score = personal_best_scores[local_best_index]
                global_best_position = personal_best_positions[local_best_index]

            if np.random.rand() < self.quantum_coeff:
                quantum_shift = np.random.uniform(-2.5, 2.5, (self.population_size, self.dim))  # More aggressive quantum leaps
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