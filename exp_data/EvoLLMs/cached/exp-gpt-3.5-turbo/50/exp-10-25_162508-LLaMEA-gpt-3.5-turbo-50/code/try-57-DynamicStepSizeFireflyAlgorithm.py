import numpy as np

class DynamicStepSizeFireflyAlgorithm(RefinedFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.step_size = 0.2
        self.step_size_min = 0.01
        self.step_size_max = 0.5
        self.step_size_factor = 1.2

    def _update_position(self, individual, best_individual):
        new_step_size = np.clip(self.step_size * np.exp(np.random.uniform(-1, 1)), self.step_size_min, self.step_size_max)
        new_position = individual + self._attractiveness(best_individual, individual) * (best_individual - individual) + new_step_size * np.random.normal(0, 1, self.dim)
        return np.clip(new_position, self.lower_bound, self.upper_bound)