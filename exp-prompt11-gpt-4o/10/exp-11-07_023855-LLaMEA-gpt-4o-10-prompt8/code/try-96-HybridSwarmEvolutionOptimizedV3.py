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
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = population[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        evaluations = self.pop_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)
            velocities *= self.inertia_weight
            velocities += self.cognitive_coef * r1 * (personal_best - population)
            velocities += self.social_coef * r2 * (global_best - population)
            population += velocities
            np.clip(population, self.bounds[0], self.bounds[1], out=population)
            
            if np.random.rand() < self.mutation_prob:
                mutation = np.random.normal(0, self.mutation_scale, (self.pop_size, self.dim))
                population += mutation
            
            scores = np.array([func(ind) for ind in population])
            evaluations += self.pop_size
            
            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best[improved] = population[improved]

            min_score_idx = scores.argmin()
            if scores[min_score_idx] < global_best_score:
                global_best_score = scores[min_score_idx]
                global_best = population[min_score_idx]

        return global_best