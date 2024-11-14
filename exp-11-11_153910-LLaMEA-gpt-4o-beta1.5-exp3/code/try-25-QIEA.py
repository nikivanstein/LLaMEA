import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(self.dim)
        self.best_global_fitness = np.inf
        self.evaluations = 0
        self.alpha = 0.1  # Learning rate for amplitude update
        self.beta = 0.5   # Probability amplitude for superposition

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_superposition(self):
        qbit_representation = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        return self.lower_bound + (self.upper_bound - self.lower_bound) * (0.5 + (qbit_representation * self.beta))

    def quantum_collapse(self, candidate):
        return candidate + self.alpha * (self.best_global_position - candidate)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            candidates = self.quantum_superposition()
            for i in range(self.pop_size):
                candidate = candidates[i]
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate = self.quantum_collapse(candidate)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                
                candidate_fitness = self.evaluate(func, candidate)
                if candidate_fitness < self.fitness[i]:
                    self.fitness[i] = candidate_fitness
                    self.population[i] = candidate
                    if candidate_fitness < self.best_global_fitness:
                        self.best_global_fitness = candidate_fitness
                        self.best_global_position = candidate

        return self.best_global_position