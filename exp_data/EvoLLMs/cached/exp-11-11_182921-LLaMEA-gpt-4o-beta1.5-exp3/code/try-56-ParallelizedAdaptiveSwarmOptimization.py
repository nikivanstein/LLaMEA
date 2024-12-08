import numpy as np
import concurrent.futures

class ParallelizedAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0

    def __call__(self, func):
        inertia_weight = 0.7
        cognitive_component = 1.5
        social_component = 1.5

        def evaluate_particle(i):
            score = func(self.population[i])
            self.func_evaluations += 1
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best[i] = self.population[i]
            return score, self.population[i]

        while self.func_evaluations < self.budget:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(evaluate_particle, range(self.population_size)))

            for score, position in results:
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = position

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_component * r1 * (self.personal_best[i] - self.population[i]) +
                                      social_component * r2 * (self.global_best_position - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            inertia_weight -= 0.001  # Reduce inertia over time

        return self.global_best_position