import numpy as np

class Enhanced_AOWO_DR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 5)
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.dynamic_scale = 0.5

    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (abs(v) ** (1 / beta))
        return 0.01 * step

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.whales[min_fitness_idx].copy()

            reduction_factor = 1 - (evaluations / self.budget)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                self.dynamic_scale = 0.5 * (1 + np.cos(np.pi * evaluations / self.budget))

                if np.random.rand() < 0.5:
                    D = np.abs(np.random.rand(self.dim) * self.best_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = self.best_solution - A * D * self.dynamic_scale
                else:
                    opp_solution = self.oppositional_solution(self.whales[i])
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = opp_solution - A * D * self.dynamic_scale

                if np.random.rand() < 0.3:
                    self.whales[i] += self.levy_flight()
                
                mutation_prob = 0.05
                if np.random.rand() < mutation_prob:
                    self.whales[i] += np.random.normal(0, 0.1, self.dim)

                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness