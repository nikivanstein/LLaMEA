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
        self.alpha = np.full((self.pop_size, self.dim), 0.5)  # Quantum probability amplitudes

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_rotation(self, alpha, beta, success):
        theta = 0.1 * (1 if success else -1)
        new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
        new_beta = alpha * np.sin(theta) + beta * np.cos(theta)
        return new_alpha, new_beta

    def observe(self, alpha):
        return np.where(np.random.rand(self.dim) < alpha, 1, -1)

    def generate_solution(self, bit_representation):
        return np.clip((bit_representation * (self.upper_bound - self.lower_bound) / 2), 
                       self.lower_bound, self.upper_bound)

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
                bit_representation = self.observe(self.alpha[i])
                candidate_solution = self.generate_solution(bit_representation)
                candidate_fitness = self.evaluate(func, candidate_solution)
                
                if candidate_fitness < self.fitness[i]:
                    success = True
                    self.population[i] = candidate_solution
                    self.fitness[i] = candidate_fitness
                    if candidate_fitness < self.best_global_fitness:
                        self.best_global_fitness = candidate_fitness
                        self.best_global_position = candidate_solution
                else:
                    success = False
                
                # Update quantum probabilities
                beta = np.sqrt(1 - self.alpha[i]**2)
                self.alpha[i], _ = self.quantum_rotation(self.alpha[i], beta, success)

        return self.best_global_position