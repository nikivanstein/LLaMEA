import numpy as np

class ImprovedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))

        for _ in range(self.budget):
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequencies = best_solution + self.alpha * (bats[i] - best_solution)
                    frequencies = np.clip(frequencies, -5.0, 5.0)
                    velocities[i] += frequencies * self.gamma
                else:
                    velocities[i] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(velocities[i])

                new_solution = np.clip(bats[i] + velocities[i], -5.0, 5.0)
                new_fitness = func(new_solution)

                if np.random.rand() < self.loudness and new_fitness < best_fitness:
                    bats[i] = new_solution
                    best_solution = new_solution
                    best_fitness = new_fitness

        return best_solution