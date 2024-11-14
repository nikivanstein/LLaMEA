import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        
        # Adaptive parameters
        self.inertia_weight = 0.5
        self.cognitive_coef = np.random.uniform(1.4, 1.6)
        self.social_coef = np.random.uniform(1.4, 1.6)
        
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, float('inf'))
        
        # Evaluate initial population
        scores = np.apply_along_axis(func, 1, particles)
        personal_best_scores = scores.copy()

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            # PSO update
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coef * r1 * (personal_best_positions - particles)
                          + self.social_coef * r2 * (global_best_position - particles))
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

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
                if trial_score < personal_best_scores[global_best_index]:
                    global_best_position = trial_vector
                    global_best_index = i

        return global_best_position