import numpy as np

class PSO_DLR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.rand(3)
                adaptive_lr_individual = 0.5 + 0.5 * (self.personal_best_fitness[i] - self.global_best_fitness) / max(1e-9, self.personal_best_fitness[i])
                adaptive_lr_collective = 0.5 + 0.5 * (self.global_best_fitness - self.personal_best_fitness[i]) / max(1e-9, self.global_best_fitness)
                adaptive_lr_random = 0.5 + 0.5 * abs(self.personal_best_fitness[i] - self.global_best_fitness)
                
                self.velocities[i] = (r1 * adaptive_lr_individual * (self.personal_best_positions[i] - self.population[i]) +
                                      r2 * adaptive_lr_collective * (self.global_best_position - self.population[i]) +
                                      r3 * adaptive_lr_random * self.velocities[i])
                
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)
                fitness = self.evaluate(func, self.population[i])
                
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position