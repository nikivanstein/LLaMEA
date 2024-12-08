import numpy as np

class EnhancedAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(20, dim * 7)
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.3  # Slightly increased cognitive coefficient
        self.social_coeff = 1.7  # Slightly decreased social coefficient
        self.local_search_prob = 0.15  # Increased local search probability
        self.eval_count = 0
        self.mutation_prob = 0.2  # New mutation probability
        self.mutation_step_size = 0.1  # Step size for mutation

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
            w = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * (self.eval_count / self.budget))
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Mutation step
            mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_prob
            mutations = np.random.normal(0, self.mutation_step_size, (self.population_size, self.dim))
            particles += mutation_mask * mutations
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

            if np.random.rand() < self.local_search_prob:
                for i in range(len(particles)):
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