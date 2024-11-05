import numpy as np

class Enhanced_AOWO_DR_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(30, self.budget // 5)
        self.population_size = self.initial_population_size
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.dynamic_scale = 0.5

    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def perturb_solution(self, solution, scale):
        perturbation = np.random.normal(0, scale, self.dim)
        return solution + perturbation

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
            scale_factor = np.exp(-evaluations / self.budget)  # Dynamic scaling factor

            self.population_size = max(5, int(self.initial_population_size * (1 - evaluations / self.budget)))
            self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

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

                self.whales[i] = self.perturb_solution(self.whales[i], scale_factor)
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness