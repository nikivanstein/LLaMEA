import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(0, 1, (self.pop_size, self.dim, 2))  # Quantum bit representation
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def quantum_to_real(self, qbit):
        return self.lower_bound + (self.upper_bound - self.lower_bound) * np.square(qbit[:, 0])

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        theta = 0.1  # Rotation angle for quantum gate

        while self.evaluations < self.budget:
            # Collapse quantum population to real values
            real_population = np.apply_along_axis(self.quantum_to_real, 2, self.population)

            # Evaluate population
            for i in range(self.pop_size):
                fitness = self.evaluate(func, real_population[i])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = real_population[i]

            # Quantum rotation gate update
            for i in range(self.pop_size):
                for j in range(self.dim):
                    if real_population[i, j] < self.best_solution[j]:
                        self.population[i, j, 0] = self.population[i, j, 0] * np.cos(theta) - self.population[i, j, 1] * np.sin(theta)
                        self.population[i, j, 1] = self.population[i, j, 0] * np.sin(theta) + self.population[i, j, 1] * np.cos(theta)
                    else:
                        self.population[i, j, 0] = self.population[i, j, 0] * np.cos(-theta) - self.population[i, j, 1] * np.sin(-theta)
                        self.population[i, j, 1] = self.population[i, j, 0] * np.sin(-theta) + self.population[i, j, 1] * np.cos(-theta)

        return self.best_solution