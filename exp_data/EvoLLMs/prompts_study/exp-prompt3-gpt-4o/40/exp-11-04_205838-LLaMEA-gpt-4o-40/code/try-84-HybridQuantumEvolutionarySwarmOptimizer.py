import numpy as np

class HybridQuantumEvolutionarySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(50, dim * 5)  # Adjust population size
        self.inertia_weight_init = 0.6
        self.inertia_weight_final = 0.3  # Reduced inertia weight for agility
        self.cognitive_coeff = 2.0  # Increased cognitive influence
        self.social_coeff = 1.5  # Reduced social influence for less crowding
        self.quantum_coeff = 0.7  # Higher probability for quantum jumps
        self.mutation_rate = 0.1  # Mutation rate for evolutionary strategy
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
            w = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * phase  # Linear inertia decay
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            if np.random.rand() < self.mutation_rate:  # Apply evolutionary mutation
                mutation_vector = np.random.normal(0, 1, (self.population_size, self.dim))
                particles += mutation_vector

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
                quantum_shift = np.random.uniform(-2.5, 2.5, (self.population_size, self.dim))  # Enhanced quantum leaps
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