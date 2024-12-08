import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_local_positions = np.copy(self.population)
        self.best_local_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = np.zeros(self.dim)
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.75  # Constriction factor
        self.beta = 0.25   # Perturbation factor

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_local_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = np.copy(self.population[i])
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Quantum-inspired position update
                p_local = self.best_local_positions[i]
                p_global = self.best_global_position
                omega = np.random.uniform(0, 1, self.dim)
                new_position = omega * p_local + (1 - omega) * p_global
                perturbation = self.beta * np.random.normal(0, 1, self.dim)
                new_position += self.alpha * perturbation
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                new_fitness = self.evaluate(func, new_position)
                
                # Update personal best
                if new_fitness < self.best_local_fitness[i]:
                    self.best_local_fitness[i] = new_fitness
                    self.best_local_positions[i] = np.copy(new_position)
                
                # Update global best
                if new_fitness < self.best_global_fitness:
                    self.best_global_fitness = new_fitness
                    self.best_global_position = np.copy(new_position)
                    
        return self.best_global_position