import numpy as np

class DynamicABC_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_step = np.full((budget, dim), 1.0)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    mutation = np.random.uniform(-1, 1, self.dim) * self.mutation_step[i]
                    trial_solution = self.population[i] + mutation * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
                        self.mutation_step[i] *= 1.1  # Increase mutation step for successful individuals
                    else:
                        self.mutation_step[i] *= 0.9  # Decrease mutation step for unsuccessful individuals
        return best_solution