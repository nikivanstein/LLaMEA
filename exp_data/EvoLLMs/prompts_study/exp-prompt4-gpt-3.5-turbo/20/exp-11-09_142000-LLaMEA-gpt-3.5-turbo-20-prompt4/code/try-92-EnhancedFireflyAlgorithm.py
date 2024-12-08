import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.5
        self.beta_min = 0.2

    def __call__(self, func):
        num_evaluations = 0
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if num_evaluations < self.budget:
                        if func(self.population[j]) < func(self.population[i]):
                            self.move_firefly(i, alpha=self.alpha, beta_min=self.beta_min)
                            num_evaluations += 1
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]