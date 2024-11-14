import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def levy_flight(self, scale=0.1):
        return np.random.standard_cauchy(size=self.dim) * scale / np.sqrt(np.abs(np.random.normal()) + 1e-12)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    if np.random.rand() < 0.1:  # 10% chance to apply Levy flight mutation
                        self.population[i] = self.population[i] + self.levy_flight()
                    else:
                        trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i])
                        if func(trial_solution) < fitness[i]:
                            self.population[i] = trial_solution
        return best_solution