import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HybridSwarmEvolutionOptimizedV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.bounds = (-5.0, 5.0)
        self.inertia_weight = 0.7298
        self.cognitive_coef = 1.49618
        self.social_coef = 1.49618
        self.mutation_prob = 0.05  # Adjusted mutation probability
        self.mutation_scale = 0.05  # Adjusted mutation scale
        self.num_threads = 4  # Adding threading for parallel evaluations

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.full(self.pop_size, np.inf)

        def evaluate_population(pop):
            return np.array([func(ind) for ind in pop])

        # Initially evaluate personal best scores
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(func, ind) for ind in population]
            for i, future in enumerate(futures):
                personal_best_scores[i] = future.result()

        global_best_idx = np.argmin(personal_best_scores)
        global_best = population[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        evaluations = self.pop_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.pop_size, self.dim)  # Moved inside loop to refresh
            mutation = np.random.normal(0, self.mutation_scale, (self.pop_size, self.dim))  # Adjusted mutation

            # Velocity and position update
            velocities *= self.inertia_weight
            velocities += self.cognitive_coef * r1 * (personal_best - population)
            velocities += self.social_coef * r2 * (global_best - population)
            population += velocities
            np.clip(population, self.bounds[0], self.bounds[1], out=population)
            
            if np.random.rand() < self.mutation_prob:
                population += mutation

            # Parallel evaluation of new scores
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(func, ind) for ind in population]
                scores = np.array([future.result() for future in futures])

            evaluations += self.pop_size
            
            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best[improved] = population[improved]

            min_score_idx = scores.argmin()
            if scores[min_score_idx] < global_best_score:
                global_best_score = scores[min_score_idx]
                global_best = population[min_score_idx]

        return global_best