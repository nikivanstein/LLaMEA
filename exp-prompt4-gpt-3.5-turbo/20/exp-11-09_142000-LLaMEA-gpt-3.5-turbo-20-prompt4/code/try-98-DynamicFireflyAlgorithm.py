import numpy as np

class DynamicFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.5
        
    def adapt_alpha(self, iteration):
        return 1 / (1 + iteration)
        
    def __call__(self, func):
        for itr in range(self.budget):
            self.alpha = self.adapt_alpha(itr)
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.move_firefly(i, self.alpha)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]