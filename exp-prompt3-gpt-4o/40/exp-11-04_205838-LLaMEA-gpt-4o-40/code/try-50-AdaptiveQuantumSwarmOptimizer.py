import numpy as np

class AdaptiveQuantumSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(40, dim * 6)  # Increased population for enhanced search space coverage
        self.inertia_weight_max = 0.8
        self.inertia_weight_min = 0.5
        self.cognitive_coeff = 2.0  # Increased cognitive coefficient for adaptive exploration
        self.social_coeff = 1.3  # Reduced social coefficient for better diversity
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
            w = self.inertia_weight_min + ((self.inertia_weight_max - self.inertia_weight_min) * (1 - np.cos(phase * np.pi / 2)))  # Adaptive inertia adjustment
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            quantum_step = np.random.normal(0, 1, (self.population_size, self.dim)) * (global_best_position - particles)
            particles += velocities + quantum_step
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

            if np.random.rand() < 0.2:  # Increased probability for local search enhancement
                neighbor_indices = np.random.choice(self.population_size, self.population_size // 5, replace=False)
                for i in neighbor_indices:
                    if self.eval_count >= self.budget:
                        break
                    neighbor = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    neighbor_score = func(neighbor)
                    self.eval_count += 1
                    if neighbor_score < personal_best_scores[i]:
                        personal_best_positions[i] = neighbor
                        personal_best_scores[i] = neighbor_score
                        if neighbor_score < global_best_score:
                            global_best_score = neighbor_score
                            global_best_position = neighbor

        return global_best_position