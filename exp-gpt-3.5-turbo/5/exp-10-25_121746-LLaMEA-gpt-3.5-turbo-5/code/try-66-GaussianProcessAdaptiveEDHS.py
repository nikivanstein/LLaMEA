import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class GaussianProcessAdaptiveEDHS(EnhancedEDHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = np.random.uniform(0.2, 0.4)  # Resetting mutation rate
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.kernel = 1.0 * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def __call__(self, func):
        for _ in range(self.budget):
            # Update mutation rate based on Gaussian process regression prediction
            if np.random.rand() < 0.05:
                self.gp.fit(self.X_train, self.y_train)
                next_rate = self.gp.predict(np.atleast_2d([self.mutation_rate]))[0]
                self.mutation_rate = np.clip(next_rate, 0.1, 0.5)
            super().__call__(func)
            self.X_train = np.append(self.X_train, [self.mutation_rate])
            self.y_train = np.append(self.y_train, func(self.get_global_best()))
        return self.get_global_best()