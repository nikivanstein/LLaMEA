import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 80  # Reduced population size for efficiency
        self.inertia_weight = 0.6  # Adaptive inertia
        self.cognitive_coef = 1.4  # Slightly tuned coefficients
        self.social_coef = 1.6
        self.mutation_factor = 0.9  # Adjusted mutation factor
        self.crossover_prob = 0.8  # Adjusted crossover probability

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)  # Simplified initialization
        
        scores = np.apply_along_axis(func, 1, particles)  # Vectorized evaluation
        personal_best_scores = np.minimum(personal_best_scores, scores)

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coef * r1 * (personal_best_positions - particles)
                          + self.social_coef * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # Differential Evolution update
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[idxs]
                mutant_vector = a + self.mutation_factor * (b - c)
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial_vector = np.where(crossover_mask, mutant_vector, particles[i])
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