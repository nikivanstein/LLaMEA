import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(20, dim * 5)  # Adapt population size
        self.inertia_weight = 0.729  # Inertia weight for PSO
        self.cognitive_coeff = 1.49445  # Cognitive coefficient for PSO
        self.social_coeff = 1.49445  # Social coefficient for PSO

    def __call__(self, func):
        eval_count = 0

        # Initialize the swarm
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        eval_count += self.population_size

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        while eval_count < self.budget:
            # PSO Update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            particles = particles + velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Evaluate new positions
            scores = np.array([func(p) for p in particles])
            eval_count += self.population_size

            # Update personal and global bests
            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = particles[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_index] < global_best_score:
                global_best_score = personal_best_scores[global_best_index]
                global_best_position = personal_best_positions[global_best_index]

            # Gradient-based exploitation
            if eval_count < self.budget - self.dim:
                for i in range(len(particles)):
                    # Approximate gradient using central difference
                    grad = np.zeros(self.dim)
                    for d in range(self.dim):
                        x_plus = np.copy(particles[i])
                        x_plus[d] += 1e-5
                        x_minus = np.copy(particles[i])
                        x_minus[d] -= 1e-5

                        grad[d] = (func(x_plus) - func(x_minus)) / (2 * 1e-5)
                        eval_count += 2

                    # Gradient descent step
                    particles[i] = particles[i] - 0.01 * grad
                    particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

        return global_best_position