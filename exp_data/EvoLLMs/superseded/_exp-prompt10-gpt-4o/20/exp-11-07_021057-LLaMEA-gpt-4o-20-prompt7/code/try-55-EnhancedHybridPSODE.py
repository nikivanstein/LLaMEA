import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 80  # Reduced population size for faster convergence
        self.inertia_weight = 0.7  # Adaptive inertia weight for better exploration
        self.cognitive_coef = 1.4
        self.social_coef = 1.6
        self.mutation_factor = 0.85  # Slightly increased mutation for diversity
        self.crossover_prob = 0.7  # Lower crossover probability for more selection pressure

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, float('inf'))

        # Compact evaluation loop
        scores = np.apply_along_axis(func, 1, particles)
        personal_best_scores = np.minimum(personal_best_scores, scores)

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            # Efficient PSO update loop
            r1, r2 = np.random.rand(self.population_size, 2).T
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coef * r1[:, np.newaxis] * (personal_best_positions - particles)
                          + self.social_coef * r2[:, np.newaxis] * (global_best_position - particles))
            particles += velocities
            np.clip(particles, self.lower_bound, self.upper_bound, out=particles)

            # Simplified DE update
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = np.random.choice(list(range(i)) + list(range(i+1, self.population_size)), 3, replace=False)
                a, b, c = particles[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, particles[i])
                np.clip(trial_vector, self.lower_bound, self.upper_bound, out=trial_vector)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < personal_best_scores[global_best_index]:
                    global_best_position = trial_vector
                    global_best_index = i

        return global_best_position