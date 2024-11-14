import numpy as np

class ImprovedFireflyAlgorithm(FireflyAlgorithm):
    def dynamic_movement(self, idx, beta_dynamic=0.8):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += beta_dynamic * np.exp(-beta_dynamic * distance) * (self.population[i] - self.population[idx])
    
    def enhanced_move_firefly(self, idx, alpha=0.5, beta_min=0.2, beta_dynamic=0.8):
        for i in range(self.budget):
            if func(self.population[i]) < func(self.population[idx]):
                distance = np.linalg.norm(self.population[idx] - self.population[i])
                self.population[idx] += alpha * np.exp(-beta_min * distance) * (1 - np.exp(-beta_dynamic * distance)) * (self.population[i] - self.population[idx])
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(self.population[j]) < func(self.population[i]):
                        self.enhanced_move_firefly(i)
        best_idx = np.argmin([func(ind) for ind in self.population])
        return self.population[best_idx]