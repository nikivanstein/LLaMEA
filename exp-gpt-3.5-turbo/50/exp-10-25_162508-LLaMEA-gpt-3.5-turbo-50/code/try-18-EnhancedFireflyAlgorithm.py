import numpy as np

class EnhancedFireflyAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.levy_coeff = 0.1

    def _generate_levy_flight(self):
        sigma = (np.math.gamma(1 + self.levy_coeff) * np.sin(np.pi * self.levy_coeff / 2) / (np.math.gamma((1 + self.levy_coeff) / 2) * self.levy_coeff * 2 ** ((self.levy_coeff - 1) / 2))) ** (1 / self.levy_coeff)
        levy = np.random.normal(0, sigma, self.dim)
        return levy

    def _update_position(self, individual, best_individual):
        levy_flight = self._generate_levy_flight()
        new_position = individual + self._attractiveness(best_individual, individual) * (best_individual - individual) + 0.01 * np.random.normal(0, 1, self.dim) + levy_flight
        return np.clip(new_position, self.lower_bound, self.upper_bound)