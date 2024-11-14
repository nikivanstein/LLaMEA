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
        self.best_personal_position = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.75  # Contraction-expansion coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_move(self, particle, mbest):
        u = np.random.uniform(0, 1, self.dim)
        beta = self.alpha * np.abs(mbest - particle)
        direction = np.random.choice([-1, 1], size=self.dim)
        return particle + direction * beta * np.log(1/u)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_personal_fitness[i] = self.fitness[i]
            self.best_personal_position[i] = self.population[i]
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            mbest = np.mean(self.best_personal_position, axis=0)
            for i in range(self.pop_size):
                new_position = self.quantum_move(self.population[i], mbest)
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                new_fitness = self.evaluate(func, new_position)
                if new_fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = new_fitness
                    self.best_personal_position[i] = new_position
                if new_fitness < self.best_global_fitness:
                    self.best_global_fitness = new_fitness
                    self.best_global_position = new_position

            self.population = np.copy(self.best_personal_position)

        return self.best_global_position