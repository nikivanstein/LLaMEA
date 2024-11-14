import numpy as np

class AC_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def adaptive_parameters(self, iter_ratio):
        w = 0.9 - 0.5 * iter_ratio
        c1 = 1.5 + 1.0 * iter_ratio
        c2 = 1.5 - 1.0 * iter_ratio
        return w, c1, c2

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.positions[i])
            self.personal_best_fitness[i] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.positions[i]
        
        while self.evaluations < self.budget:
            iter_ratio = self.evaluations / self.budget
            w, c1, c2 = self.adaptive_parameters(iter_ratio)
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = w * self.velocities[i] + \
                                    c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + \
                                    c2 * r2 * (self.global_best_position - self.positions[i])
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)
                
                fitness = self.evaluate(func, self.positions[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_positions[i] = self.positions[i]
                    self.personal_best_fitness[i] = fitness
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.positions[i]

        return self.global_best_position