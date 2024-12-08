import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.beta = 0.5  # Contraction-expansion coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                p_best = self.personal_best[i]
                g_best = self.global_best_position
                u = np.random.rand(self.dim)
                mbest = np.mean(self.population, axis=0)
                fi = np.random.rand()

                p = fi * p_best + (1 - fi) * g_best
                temp = np.abs(p - self.population[i])
                L = np.random.choice([-1, 1], size=self.dim)

                self.population[i] = p + L * self.beta * temp * np.log(1/u)

                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.population[i]
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.population[i]

        return self.global_best_position