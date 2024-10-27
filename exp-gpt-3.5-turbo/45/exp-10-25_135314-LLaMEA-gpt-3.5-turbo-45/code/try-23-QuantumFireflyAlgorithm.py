import numpy as np

class QuantumFireflyAlgorithm:
    def __init__(self, budget, dim, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def quantum_rotation(self, current, best, theta):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(rotation_matrix, current - best) + best

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        light_intensities = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if light_intensities[j] > light_intensities[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness_ij = self.alpha * np.exp(-self.gamma * distance**2)
                        rotation_angle = attractiveness_ij * np.pi
                        population[i] = self.quantum_rotation(population[i], population[j], rotation_angle)
                        light_intensities[i] = func(population[i])

        return population[np.argmin(light_intensities)]