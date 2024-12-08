import numpy as np

class AdaptiveChaosDrivenFirefly:
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
        self.alpha = 0.5  # Randomness factor
        self.beta_min = 0.2  # Minimum attraction
        self.gamma = 1.0  # Absorption coefficient

        # Chaos sequence to control randomness
        self.chaos_sequence = np.random.rand(self.budget)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if func(self.population[j]) < func(self.population[i]):
                        r = np.linalg.norm(self.population[i] - self.population[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r**2)
                        step = beta * (self.population[j] - self.population[i])
                        step += self.alpha * (self.chaos_sequence[self.func_evaluations] - 0.5)
                        new_position = self.population[i] + step
                        new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                        # Evaluate new candidate
                        new_score = func(new_position)
                        self.func_evaluations += 1
                        if new_score < self.best_score:
                            self.best_score = new_score
                            self.best_position = new_position
                        if new_score < func(self.population[i]):
                            new_population[i] = new_position

            self.population = new_population

            # Adaptive alpha using chaos
            self.alpha = 0.5 * (1 + self.chaos_sequence[self.func_evaluations % self.budget])

        return self.best_position