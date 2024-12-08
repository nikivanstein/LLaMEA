import numpy as np

class EnhancedDynamicABC:
    def __init__(self, budget, dim, elitism_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.elitism_rate = elitism_rate
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution

            # Elitism: Preserve the best solutions found
            num_elites = int(self.elitism_rate * self.budget)
            elite_indices = np.argsort(fitness)[:num_elites]
            for idx in elite_indices:
                self.population[idx] = best_solution
        return best_solution