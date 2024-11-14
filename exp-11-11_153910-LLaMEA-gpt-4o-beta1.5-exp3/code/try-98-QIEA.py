import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.q_population = np.random.rand(self.pop_size, self.dim)  # Quantum bit probabilities
        self.best_sol = None
        self.best_fit = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def collapse(self):
        # Collapse quantum bits to real solutions based on probabilities
        return np.where(self.q_population > 0.5, self.upper_bound, self.lower_bound)

    def update_quantum_bits(self, xi, best):
        # Update quantum bits towards the best solution
        for d in range(self.dim):
            if np.random.rand() < 0.5:
                self.q_population[xi, d] = self.q_population[xi, d] + 0.1 * (best[d] - self.q_population[xi, d])
            else:
                self.q_population[xi, d] = self.q_population[xi, d] + 0.1 * (self.q_population[xi, d] - best[d])

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        real_population = self.collapse()
        for i in range(self.pop_size):
            fitness = self.evaluate(func, real_population[i])
            if fitness < self.best_fit:
                self.best_fit = fitness
                self.best_sol = real_population[i]

        while self.evaluations < self.budget:
            real_population = self.collapse()
            for i in range(self.pop_size):
                fitness = self.evaluate(func, real_population[i])
                if fitness < self.best_fit:
                    self.best_fit = fitness
                    self.best_sol = real_population[i]

                # Update quantum bits
                self.update_quantum_bits(i, self.best_sol)
        
        return self.best_sol