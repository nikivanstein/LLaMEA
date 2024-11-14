import numpy as np

class EnhancedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.2  # Alpha value for elite selection

    def levy_update(self, x):
        step = self.levy_flight()
        new_x = x + step * np.random.normal(0, 1, self.dim)
        new_x = np.clip(new_x, self.lb, self.ub)
        return new_x

    def elite_selection(self, x, new_x, func):
        fx = func(x)
        new_fx = func(new_x)
        if new_fx < fx:
            return new_x
        else:
            if np.random.rand() < self.alpha:
                return new_x
            else:
                return x

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i != j:
                        x = self.population[i]
                        y = self.population[j]
                        new_x = self.move_firefly(x, y)
                        new_x = self.levy_update(new_x)
                        self.population[i] = self.elite_selection(x, new_x, func)
        return np.min([func(x) for x in self.population])