import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.initial_population_size = self.population_size
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([float('inf')] * self.population_size)
        
        # Evaluate initial population
        for i in range(self.population_size):
            score = func(particles[i])
            personal_best_scores[i] = score

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            # Adaptive inertia weight
            self.inertia_weight = max(self.inertia_weight_min, self.inertia_weight * 0.99)
            
            # PSO update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coef * r1 * (personal_best_positions[i] - particles[i])
                                 + self.social_coef * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            # Dynamic population reduction
            if self.budget - eval_count < self.initial_population_size:
                self.population_size = max(2, self.budget - eval_count)

            # Differential Evolution update with crossover
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = a + self.mutation_factor * (b - c)
                trial_vector = np.copy(particles[i])
                
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]
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