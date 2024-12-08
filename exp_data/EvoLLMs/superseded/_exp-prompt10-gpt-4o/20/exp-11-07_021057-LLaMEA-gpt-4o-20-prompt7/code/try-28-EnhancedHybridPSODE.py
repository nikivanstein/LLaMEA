import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 80  # Reduced size for quicker evaluations
        self.inertia_weight = 0.7  # Adaptive inertia weight
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.9  # Slightly higher mutation factor
        self.crossover_prob = 0.8  # Adjusted crossover probability

    def __call__(self, func):
        # Initialize particles with slight improvement
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))  # Simplified initialization
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        # Evaluate initial population
        scores = np.apply_along_axis(func, 1, particles)
        personal_best_scores = np.minimum(personal_best_scores, scores)

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            # PSO update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coef * r1 * (personal_best_positions[i] - particles[i])
                                 + self.social_coef * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            # Differential Evolution update with crossover
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
                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score

            # Adaptive adjustment to parameters
            self.inertia_weight *= 0.99  # Gradually reduce inertia weight for convergence

            # Early stopping if no improvement
            if eval_count > self.budget * 0.8 and np.isclose(global_best_score, np.min(personal_best_scores), atol=1e-8):
                break

        return global_best_position