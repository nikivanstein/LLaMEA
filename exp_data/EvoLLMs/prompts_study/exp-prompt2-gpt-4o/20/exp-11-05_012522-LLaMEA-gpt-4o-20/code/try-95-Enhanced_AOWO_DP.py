import numpy as np

class Enhanced_AOWO_DP:
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

    def reduce_dimensionality(self, solution, factor):
        mask = np.random.rand(self.dim) < factor
        reduced_solution = solution.copy()
        reduced_solution[mask] = self.best_solution[mask] if self.best_solution is not None else 0
        return reduced_solution

    def differential_perturbation(self, solution):
        r1, r2 = np.random.choice(self.population_size, 2, replace=False)
        return solution + 0.8 * (self.whales[r1] - self.whales[r2])

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

            # Adaptive population size
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

                # New differential perturbation step
                self.whales[i] = self.differential_perturbation(self.whales[i])
                
                self.whales[i] = self.reduce_dimensionality(self.whales[i], reduction_factor)
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness