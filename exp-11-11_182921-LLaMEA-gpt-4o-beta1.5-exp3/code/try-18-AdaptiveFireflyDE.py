import numpy as np

class AdaptiveFireflyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.5  # Initial attractiveness of fireflies
        self.beta_min = 0.2
        self.gamma = 1.0  # Absorption coefficient

    def __call__(self, func):
        scores = np.array([func(ind) for ind in self.population])
        self.best_position = self.population[np.argmin(scores)]
        self.best_score = np.min(scores)
        self.func_evaluations += self.population_size

        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if scores[i] > scores[j]:
                        r = np.linalg.norm(self.population[i] - self.population[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r**2)
                        step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        self.population[i] = np.clip(self.population[i] * (1.0 - beta) + self.population[j] * beta + step,
                                                     self.lower_bound, self.upper_bound)

                # Differential Evolution / Best / 2 Mutation
                indices = np.random.choice(self.population_size, 5, replace=False)
                x_best, x1, x2, x3, x4 = self.population[np.argmin(scores)], self.population[indices[1]], self.population[indices[2]], self.population[indices[3]], self.population[indices[4]]
                mutant_vector = x_best + self.F * (x1 - x2 + x3 - x4)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                # Selection
                trial_score = func(trial_vector)
                self.func_evaluations += 1

                if trial_score < scores[i]:
                    self.population[i] = trial_vector
                    scores[i] = trial_score

                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = trial_vector

            self.alpha *= 0.98  # Gradually decrease attractiveness

        return self.best_position