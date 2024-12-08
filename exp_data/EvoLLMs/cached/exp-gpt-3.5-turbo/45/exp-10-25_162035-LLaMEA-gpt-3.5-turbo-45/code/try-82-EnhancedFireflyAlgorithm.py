import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 0.8
        self.alpha = 0.1
        self.cr = 0.5  # Crossover rate for DE
        self.f = 0.5  # Differential weight for DE

    def differential_evolution(self, current, target, t):
        mutant = current + self.f * (self.population[target] - self.population[current])
        crossover_mask = np.random.rand(self.dim) < self.cr
        trial = np.where(crossover_mask, mutant, current)
        return np.clip(trial, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        attractiveness_ij = self.attractiveness(self.population[i], self.population[j])
                        
                        # Incorporate differential evolution for updating position
                        self.population[i] = self.differential_evolution(self.population[i], j, t) * attractiveness_ij
                        
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]