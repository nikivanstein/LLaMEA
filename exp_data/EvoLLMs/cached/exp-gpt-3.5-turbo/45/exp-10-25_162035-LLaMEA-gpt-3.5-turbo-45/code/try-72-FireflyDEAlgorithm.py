import numpy as np

class FireflyDEAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 0.8
        self.alpha = 0.1
        self.cr = 0.5
        self.f = 0.5

    def differential_evolution(self, t):
        for i in range(self.budget):
            for j in range(self.budget):
                k, l, m = np.random.choice(self.budget, 3, replace=False)
                trial_vector = self.population[k] + self.f * (self.population[l] - self.population[m])
                for d in range(self.dim):
                    if np.random.rand() > self.cr:
                        trial_vector[d] = self.population[i][d]
                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector

    def __call__(self, func):
        for t in range(self.budget):
            self.differential_evolution(t)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness_ij = self.attractiveness(self.population[i], self.population[j])
                        self.population[i] = self.move_firefly(self.population[i], self.population[j], t) * attractiveness_ij
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]