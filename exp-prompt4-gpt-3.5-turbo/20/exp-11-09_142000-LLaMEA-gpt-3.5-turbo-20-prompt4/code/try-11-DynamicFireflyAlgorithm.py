import numpy as np

class DynamicFireflyAlgorithm(FireflyAlgorithm):
    def attractiveness(self, light_intensity, distance):
        gamma = 0.1
        beta = 1.0 / (1.0 + gamma * np.mean(np.abs(self.population)))
        return light_intensity / (1 + beta * distance)