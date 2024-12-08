import numpy as np

class FireflyDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9, alpha=0.5, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        pbest = population.copy()
        pbest_vals = np.array([func(ind) for ind in pbest])
        gbest = pbest[np.argmin(pbest_vals)]
        gbest_val = np.min(pbest_vals)

        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(population[j]) < func(population[i]):
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        population[i] += beta * (population[j] - population[i])

                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
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

            # Adaptive parameters
            self.f = np.clip(self.f + self.alpha * np.random.randn(), 0.1, 0.9)
            self.cr = np.clip(self.cr + self.alpha * np.random.randn(), 0.1, 0.9)

        return gbest