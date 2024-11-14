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
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.evaluations = 0
        self.beta = 0.5  # Contraction-expansion coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_position(self, position, personal_best, global_best):
        phi = np.random.rand(self.dim)
        mbest = np.mean(self.population, axis=0)
        u = np.random.rand()
        new_position = phi * (personal_best - abs(position - mbest)) + (1 - phi) * (global_best - abs(position - mbest))
        return new_position + np.log(1 / u) * np.sign(global_best - position) * self.beta

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]
            self.best_personal_fitness[i] = self.fitness[i]
            self.best_personal_positions[i] = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.population[i] = self.update_position(self.population[i], 
                                                          self.best_personal_positions[i], 
                                                          self.best_global_position)
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
                
                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = current_fitness
                    self.best_personal_positions[i] = self.population[i]
                    if current_fitness < self.best_global_fitness:
                        self.best_global_fitness = current_fitness
                        self.best_global_position = self.population[i]

        return self.best_global_position