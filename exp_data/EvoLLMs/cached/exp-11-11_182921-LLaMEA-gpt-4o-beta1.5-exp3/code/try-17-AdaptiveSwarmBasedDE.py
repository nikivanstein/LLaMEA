import numpy as np

class AdaptiveSwarmBasedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.func_evaluations = 0
        self.best_global_score = float('inf')
        self.best_global_position = None
        self.personal_best = self.population.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        self.best_global_position = self.initialize_population(func)

        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover: DE/current-to-best/1 strategy
                indices = np.random.choice(self.population_size, 4, replace=False)
                x0, x1, x2, x3 = self.population[i], self.population[indices[0]], self.population[indices[1]], self.population[indices[2]]
                mutant_vector = x0 + self.F * (self.best_global_position - x0) + self.F * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                # Crossover operation
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                
                # Evaluate new solution
                trial_score = func(trial_vector)
                self.func_evaluations += 1
                
                # Selection
                if trial_score < func(self.population[i]):
                    self.population[i] = trial_vector

                # Update personal bests
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector

                    # Update global best
                    if trial_score < self.best_global_score:
                        self.best_global_score = trial_score
                        self.best_global_position = trial_vector

            # Adaptive adjustment of F and CR based on success rate
            success_rate = np.mean(self.personal_best_scores < self.best_global_score)
            self.F = 0.4 + 0.2 * success_rate
            self.CR = 0.8 + 0.1 * (1 - success_rate)

        return self.best_global_position

    def initialize_population(self, func):
        for i in range(self.population_size):
            score = func(self.population[i])
            self.func_evaluations += 1
            if score < self.best_global_score:
                self.best_global_score = score
                self.best_global_position = self.population[i]
            self.personal_best_scores[i] = score
            self.personal_best[i] = self.population[i]
        return self.best_global_position