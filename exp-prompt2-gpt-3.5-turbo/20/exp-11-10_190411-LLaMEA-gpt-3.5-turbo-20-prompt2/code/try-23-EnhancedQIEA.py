import numpy as np

class EnhancedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 0.1

    def adaptive_step_size(self, iteration):
        return max(0.01, min(1.0, 1.0 - iteration / self.budget))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_solution = population[np.argmin(func(population))]

        for i in range(self.budget):
            theta = np.random.uniform(-np.pi, np.pi, size=(self.dim))
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)])
            population = np.dot(population, rotation_matrix)

            sigma = self.adaptive_step_size(i) * self.sigma
            offspring = population + np.random.normal(0, sigma, size=population.shape)
            population = np.where(func(offspring) < func(population), offspring, population)

            current_best_solution = population[np.argmin(func(population))]
            if func(current_best_solution) < func(best_solution):
                best_solution = current_best_solution

        return best_solution