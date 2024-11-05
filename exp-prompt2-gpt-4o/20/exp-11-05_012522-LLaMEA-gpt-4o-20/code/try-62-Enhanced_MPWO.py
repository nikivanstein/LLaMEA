import numpy as np

class Enhanced_MPWO:
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

    def oppositional_solution(self, solution, elite):
        return self.lower_bound + self.upper_bound - solution + 0.2 * (elite - solution)

    def __call__(self, func):
        evaluations = 0
        phase_switch = self.budget // 3

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

                if evaluations < phase_switch:
                    random_offset = np.random.normal(0, 1, self.dim)
                    self.whales[i] = self.best_solution + self.dynamic_scale * random_offset
                else:
                    if np.random.rand() < 0.5:
                        D = np.abs(np.random.rand(self.dim) * self.best_solution - self.whales[i])
                        A = 2 * np.random.rand(self.dim) - 1
                        self.whales[i] = self.best_solution - A * D * self.dynamic_scale
                    else:
                        elite = self.whales[np.random.randint(self.population_size)]
                        opp_solution = self.oppositional_solution(self.whales[i], elite)
                        D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                        A = 2 * np.random.rand(self.dim) - 1
                        self.whales[i] = opp_solution - A * D * self.dynamic_scale

                self.whales[i] *= reduction_factor
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness