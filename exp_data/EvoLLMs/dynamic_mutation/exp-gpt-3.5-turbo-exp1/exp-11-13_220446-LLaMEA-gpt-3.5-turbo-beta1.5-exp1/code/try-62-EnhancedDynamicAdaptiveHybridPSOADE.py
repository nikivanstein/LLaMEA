import numpy as np

class EnhancedDynamicAdaptiveHybridPSOADE(DynamicAdaptiveHybridPSOADE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.chaos_map = np.array([0.4, 0.7, 0.9, 0.1, 0.3])  # Chaotic map parameters
        
    def chaotic_map(self, x, iteration):
        dynamic_chaos_map = np.array([0.4 + np.sin(0.1*iteration), 0.7 + np.cos(0.05*iteration), 0.9 - np.tanh(0.02*iteration), 0.1 + np.sin(0.03*iteration), 0.3 - np.cos(0.08*iteration)]) + np.random.uniform(-0.05, 0.05, 5)
        return np.mod((dynamic_chaos_map[0] * x * (1 - x) + dynamic_chaos_map[1] * np.sin(np.pi * x)) * (1 - x), 1.0)