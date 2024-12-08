import numpy as np

class ImprovedDynamicInertiaWeightEnhancedSocialSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def _local_search(self, x, f):
        step_size = np.ones(self.dim)
        for _ in range(10):
            x_new = x + step_size * np.random.normal(size=self.dim)
            improvements = f(x_new) < f(x)
            x = np.where(improvements, x_new, x)
            step_size = np.where(improvements, step_size * 1.1, step_size * 0.9)  # Adaptive step adjustment per dimension
        return x