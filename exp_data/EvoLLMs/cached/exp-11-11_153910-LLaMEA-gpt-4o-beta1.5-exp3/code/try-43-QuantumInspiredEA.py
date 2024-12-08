import numpy as np

class QuantumInspiredEA:
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
        self.alpha = 0.5  # Quantum rotation angle

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_rotation(self, xi, x_best):
        # Quantum-inspired rotation to create new candidate solutions
        q_bit = np.random.uniform(0, 1, self.dim)
        quantum_vector = self.alpha * (x_best - xi) + (1 - self.alpha) * q_bit
        return xi + quantum_vector

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
                quantum_candidate = self.quantum_rotation(self.population[i], self.best_global_position)
                quantum_candidate = np.clip(quantum_candidate, self.lower_bound, self.upper_bound)
                
                candidate_fitness = self.evaluate(func, quantum_candidate)
                if candidate_fitness < self.fitness[i]:
                    self.population[i] = quantum_candidate
                    self.fitness[i] = candidate_fitness
                    if candidate_fitness < self.best_global_fitness:
                        self.best_global_fitness = candidate_fitness
                        self.best_global_position = quantum_candidate

        return self.best_global_position