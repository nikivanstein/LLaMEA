import numpy as np

class EnhancedAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.3, 0.7
        self.F_adapt, self.CR_adapt = 0.1, 0.1
        
    def adapt_control_parameters(self, current):
        self.F = np.clip(self.F * np.exp(self.F_adapt * np.random.normal()), self.F_min, self.F_max)
        self.CR = np.clip(self.CR * np.exp(self.CR_adapt * np.random.normal()), self.CR_min, self.CR_max)
        
    def __call__(self, func):
        return super().__call__(func)