import numpy as np

class EnhancedDynamicInertiaWeightEnhancedSocialSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def _local_search(self, x, f):
        step_size = 0.5
        for _ in range(10):
            x_new = x + step_size * np.random.normal(size=self.dim)
            if f(x_new) < f(x):
                x = x_new
                step_size *= 1.2  # Adaptive step adjustment for faster convergence
            else:
                step_size *= 0.8  # Adaptive step adjustment for faster convergence
        return x