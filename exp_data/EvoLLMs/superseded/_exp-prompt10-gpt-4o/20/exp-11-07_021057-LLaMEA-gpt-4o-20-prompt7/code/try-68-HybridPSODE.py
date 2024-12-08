import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.inertia_weight = 0.7  # Increased for better exploration
        self.cognitive_coef = 1.2  # Adjusted for convergence
        self.social_coef = 1.8  # Slightly increased for social learning
        self.mutation_factor = 0.7  # Slightly reduced for diversity
        self.crossover_prob = 0.85  # Adjusted for exploration-exploitation balance

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))  # Non-zero initial velocities
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, float('inf'))  # Simplified initialization

        global_best_index = 0
        eval_count = 0

        while eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, particles)  # Vectorized function evaluation
            eval_count += self.population_size

            improved = scores < personal_best_scores  # Identify improvements
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = particles[improved]

            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index]

            if eval_count >= self.budget:
                break

            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coef * r1 * (personal_best_positions - particles)
                          + self.social_coef * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.mutation_factor * (b - c)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, particles[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                    if trial_score < personal_best_scores[global_best_index]:
                        global_best_position = trial_vector
                        global_best_index = i

        return global_best_position