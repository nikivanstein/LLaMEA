import numpy as np

class NovelAdaptiveMetaheuristicAlgorithm:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        pbest = population.copy()
        pbest_vals = np.array([func(ind) for ind in pbest])
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)

        for _ in range(self.budget):
            for i in range(self.pop_size):
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                trial = np.clip(pbest[a] + self.f * (pbest[b] - pbest[c]), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.cr
                population[i] = np.where(crossover, trial, population[i])

                if func(population[i]) < pbest_vals[i]:
                    pbest[i] = population[i]
                    pbest_vals[i] = func(population[i])

            new_gbest_val = np.min(pbest_vals)
            if new_gbest_val < gbest_val:
                gbest = pbest[np.argmin(pbest_vals)]
                gbest_val = new_gbest_val

            # Adaptive mutation rates and strategies based on optimization progress
            self.f = 0.1 if np.random.rand() < 0.2 else 0.5
            self.cr = 0.1 if np.random.rand() < 0.2 else 0.9

        return gbest