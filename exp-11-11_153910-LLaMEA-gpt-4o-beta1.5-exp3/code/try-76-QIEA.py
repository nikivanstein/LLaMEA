import numpy as np

class QIEA:
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
        self.alpha = np.pi / 4  # Quantum rotation angle
        
    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)
    
    def quantum_rotation(self, q, b):
        new_q = q.copy()
        for i in range(self.dim):
            theta = np.arctan2(new_q[i, 1], new_q[i, 0])
            delta_theta = self.alpha if q[i, 0] * b[i] < 0 else -self.alpha
            new_theta = theta + delta_theta
            new_q[i, 0] = np.cos(new_theta)
            new_q[i, 1] = np.sin(new_theta)
        return new_q
    
    def collapse_to_solution(self, q):
        return np.sign(q[:, 0]) * self.upper_bound * (np.abs(q[:, 0]) / np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2))
    
    def __call__(self, func):
        np.random.seed(42)
        
        # Quantum population representation: each position is a pair (cos(theta), sin(theta))
        quantum_population = np.random.uniform(-1, 1, (self.pop_size, self.dim, 2))
        quantum_population = quantum_population / np.linalg.norm(quantum_population, axis=-1, keepdims=True)
        
        # Initial evaluation
        for i in range(self.pop_size):
            solution = self.collapse_to_solution(quantum_population[i])
            self.fitness[i] = self.evaluate(func, solution)
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = solution
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                solution = self.collapse_to_solution(quantum_population[i])
                trial_fitness = self.evaluate(func, solution)
                
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_global_fitness:
                        self.best_global_fitness = trial_fitness
                        self.best_global_position = solution
                        
                # Quantum rotation
                quantum_population[i] = self.quantum_rotation(quantum_population[i], self.best_global_position)
        
        return self.best_global_position