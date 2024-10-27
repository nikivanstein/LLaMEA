import numpy as np

class AdaptiveDECSAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.F = 0.5
        self.CR = 0.3
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def levy_flight(self, x, best, step_size=0.01):
        return x + step_size * np.random.standard_cauchy(size=len(x)) * (x - best)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        pbest = population.copy()
        gbest = pbest[np.argmin([func(p) for p in pbest])

        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                idxs = np.random.choice(len(population), 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound) if np.random.rand() < self.CR else population[i]
                trial = self.levy_flight(mutant, gbest)

                if func(trial) < func(population[i]):
                    population[i] = trial
                    pbest[i] = trial
                    if func(trial) < func(gbest):
                        gbest = trial

        return gbest