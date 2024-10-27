import numpy as np

class RefinedHarmonySearch(HarmonySearch):
    def __call__(self, func):
        for t in range(self.budget):
            new_solution = np.zeros((1, self.dim))
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[0, j] = self.harmonies[np.random.randint(self.budget), j]
                else:
                    new_solution[0, j] = self.lower_bound + np.random.rand() * (self.upper_bound - self.lower_bound)
                if np.random.rand() < self.par:
                    new_solution[0, j] += np.random.uniform(-self.bandwidth, self.bandwidth)
                if np.random.rand() < 0.35:  # Probability of 0.35 to refine individual lines
                    new_solution[0, j] = self.harmonies[np.argmin(self.fitness), j] + np.random.uniform(-self.bandwidth, self.bandwidth)

            new_fitness = func(new_solution)
            if new_fitness < np.max(self.fitness):
                self.fitness[np.argmax(self.fitness)] = new_fitness
                self.harmonies[np.argmax(self.fitness)] = new_solution
        return self.harmonies[np.argmin(self.fitness)]