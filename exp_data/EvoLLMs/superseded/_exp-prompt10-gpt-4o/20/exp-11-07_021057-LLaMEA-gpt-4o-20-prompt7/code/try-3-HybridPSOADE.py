import numpy as np

class HybridPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Reduced to enhance speed
        self.inertia_weight = 0.5
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, float('inf'))

        scores = np.apply_along_axis(func, 1, particles)  # Vectorized evaluation
        eval_count = self.population_size
        personal_best_scores = scores.copy()
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        while eval_count < self.budget:
            # Adaptive PSO update
            inertia_weight = 0.9 - (0.5 * eval_count / self.budget)
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (inertia_weight * velocities[i]
                                 + self.cognitive_coef * r1 * (personal_best_positions[i] - particles[i])
                                 + self.social_coef * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution update
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant_vector, particles[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

            # Update global best
            current_global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_index] < personal_best_scores[global_best_index]:
                global_best_position = personal_best_positions[current_global_best_index]
                global_best_index = current_global_best_index

        return global_best_position