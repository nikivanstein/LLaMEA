import numpy as np

class RefinedDynamicPopSizeLevyFireflyAlgorithm(DynamicPopSizeLevyFireflyAlgorithm):
    def crowding_distance(self, population):
        distances = np.zeros(len(population))
        for i in range(self.pop_size):
            distances[i] = np.sum(np.sqrt(np.sum(np.square(population - population[i]), axis=1)))
        return distances

    def levy_update_crowding(self, x, population):
        step = self.levy_flight()
        new_x = x + step * np.random.normal(0, 1, self.dim)
        distances = self.crowding_distance(population)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        normalized_dist = (distances - min_dist) / (max_dist - min_dist)
        crowding_factor = np.exp(-normalized_dist)
        new_x = x + crowding_factor * (new_x - x)
        return np.clip(new_x, self.lb, self.ub)