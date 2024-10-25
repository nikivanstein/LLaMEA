import numpy as np

class ImprovedEnhancedHybridDynamicAlgorithm(EnhancedHybridDynamicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.probability = 0.07317073170731707