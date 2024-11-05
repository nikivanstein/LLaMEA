import numpy as np

class RefinedEnhancedDynamicDE(EnhancedDynamicDE):
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
        trial = np.where(cross_points, mutant, target)
        return trial