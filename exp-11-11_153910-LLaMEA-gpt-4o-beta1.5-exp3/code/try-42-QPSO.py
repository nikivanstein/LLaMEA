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
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.5  # Contraction-expansion coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_behavior(self, particle, global_best, local_best, beta):
        u = np.random.uniform(size=self.dim)
        mbest = (global_best + local_best) / 2.0
        return mbest + beta * np.abs(global_best - particle) * np.log(1.0 / u)
        
    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_local_positions[i] = self.population[i]
            self.best_local_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            beta = self.alpha * (1 - self.evaluations / self.budget)  # Decreasing beta over time
            for i in range(self.pop_size):
                new_position = self.quantum_behavior(self.population[i], self.best_global_position, self.best_local_positions[i], beta)
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                new_fitness = self.evaluate(func, new_position)
                if new_fitness < self.best_local_fitness[i]:
                    self.best_local_positions[i] = new_position
                    self.best_local_fitness[i] = new_fitness
                    if new_fitness < self.best_global_fitness:
                        self.best_global_fitness = new_fitness
                        self.best_global_position = new_position

                self.population[i] = new_position

        return self.best_global_position