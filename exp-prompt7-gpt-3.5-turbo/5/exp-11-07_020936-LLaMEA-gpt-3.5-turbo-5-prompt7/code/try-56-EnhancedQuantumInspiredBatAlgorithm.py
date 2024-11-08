import numpy as np

class EnhancedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def __call__(self, func):
        bats = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution, best_fitness = bats[0].copy(), func(bats[0])

        for _ in range(self.budget):
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequencies = np.clip(best_solution + self.alpha * (bats[i] - best_solution), self.lower_bound, self.upper_bound)
                    velocities[i] += frequencies * self.gamma
                else:
                    if np.linalg.norm(velocities[i]) != 0:
                        velocities[i] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(velocities[i])
                    else:
                        velocities[i] = np.random.uniform(-1, 1, self.dim)

                bats[i] += velocities[i]
                np.clip(bats[i], self.lower_bound, self.upper_bound, out=bats[i])
                new_fitness = func(bats[i])

                if np.random.rand() < self.loudness and new_fitness < best_fitness:
                    best_solution, best_fitness = bats[i].copy(), new_fitness

        return best_solution