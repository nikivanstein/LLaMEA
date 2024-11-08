import numpy as np

class HybridSwarmEvolutionOptimizedV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.bounds = (-5.0, 5.0)
        self.inertia_weight = 0.7298
        self.cognitive_coef = 1.49618
        self.social_coef = 1.49618
        self.mutation_prob = 0.1
        self.mutation_scale = 0.1

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(*self.bounds, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.apply_along_axis(func, 1, population)
        global_best_idx = personal_best_scores.argmin()
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)
            
            # Update velocity and position
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coef * r1 * (personal_best - population) +
                          self.social_coef * r2 * (global_best - population))
            population += velocities
            np.clip(population, *self.bounds, out=population)

            # Apply mutation with vectorized operation
            mutation_mask = np.random.rand(self.pop_size, self.dim) < self.mutation_prob
            mutation = np.random.normal(0, self.mutation_scale, (self.pop_size, self.dim))
            population += mutation_mask * mutation
            
            # Evaluate population
            scores = np.apply_along_axis(func, 1, population)
            evaluations += self.pop_size

            # Update personal bests
            improved = scores < personal_best_scores
            personal_best_scores = np.where(improved, scores, personal_best_scores)
            personal_best[improved] = population[improved]

            # Update global best
            min_score_idx = scores.argmin()
            if scores[min_score_idx] < global_best_score:
                global_best_score = scores[min_score_idx]
                global_best = population[min_score_idx]

        return global_best