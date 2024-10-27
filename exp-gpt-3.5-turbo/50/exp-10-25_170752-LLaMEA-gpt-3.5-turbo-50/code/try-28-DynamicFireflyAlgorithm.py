# import numpy as np

class DynamicFireflyAlgorithm(DynamicFireflyAlgorithm):
    def adapt_parameters(self, iter_count):
        self.alpha = max(0.2, self.alpha * (1 - iter_count / self.budget))
        self.gamma = min(1.0, self.gamma + iter_count / (2 * self.budget))
