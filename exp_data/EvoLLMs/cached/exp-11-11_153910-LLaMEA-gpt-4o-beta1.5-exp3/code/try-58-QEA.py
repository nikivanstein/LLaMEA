import numpy as np

class QEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = np.random.rand(self.pop_size, self.dim)  # Probability amplitude
        self.beta = np.sqrt(1 - self.alpha ** 2)  # Orthogonal probability amplitude

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_rotation(self, index, best_position):
        for d in range(self.dim):
            if np.random.rand() < self.alpha[index, d]:
                self.population[index, d] = best_position[d]
            else:
                self.population[index, d] += np.random.normal(0, 1) * (self.upper_bound - self.lower_bound) / 10.0

    def update_quantum_states(self, index):
        self.alpha[index] = np.random.rand(self.dim)
        self.beta[index] = np.sqrt(1 - self.alpha[index] ** 2)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.quantum_rotation(i, self.best_global_position)
                trial_fitness = self.evaluate(func, self.population[i])
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_global_fitness:
                        self.best_global_fitness = trial_fitness
                        self.best_global_position = self.population[i]
                else:
                    self.update_quantum_states(i)

        return self.best_global_position