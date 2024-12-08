import numpy as np

class FireflyAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def attractiveness(self, light_intensity, distance):
        beta = 1
        return light_intensity / (1 + beta * distance)

    def move_firefly(self, idx, alpha=0.5, beta_min=0.2):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (self.population[i] - self.population[idx])
    
    def differential_evolution(self, idx, f=0.5, cr=0.7):
        candidates = [i for i in range(self.budget) if i != idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = self.population[a] + f * (self.population[b] - self.population[c])
        crossover = np.random.rand(self.dim) < cr
        trial = np.where(crossover, mutant, self.population[idx])
        if func(trial) < func(self.population[idx]):
            self.population[idx] = trial

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i)
                    else:
                        self.differential_evolution(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]