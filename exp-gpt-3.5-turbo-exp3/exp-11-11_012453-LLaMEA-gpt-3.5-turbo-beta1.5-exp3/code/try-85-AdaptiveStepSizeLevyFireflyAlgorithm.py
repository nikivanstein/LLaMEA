import numpy as np

class AdaptiveStepSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 1.5  # Levy flight step size exponent

    def adaptive_step_size(self, x, f_best, f_worst):
        step_size = np.abs(f_best - f_worst) / np.linalg.norm(x - self.best_solution)
        return step_size

    def levy_update(self, x, f_best, f_worst):
        step_size = self.adaptive_step_size(x, f_best, f_worst)
        step = self.levy_flight() * step_size
        new_x = x + step * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, self.lb, self.ub)