import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.beta = 1.5  # Constriction factor

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_position_update(self, particle, pbest, gbest):
        u = np.random.uniform(0, 1, self.dim)
        mbest = (pbest + gbest) / 2
        return mbest + self.beta * np.abs(pbest - gbest) * np.log(1 / u)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.pbest_fitness[i] = fitness
            self.pbest_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                new_position = self.quantum_position_update(
                    self.population[i], 
                    self.pbest_positions[i], 
                    self.global_best_position
                )
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                new_fitness = self.evaluate(func, new_position)
                if new_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = new_fitness
                    self.pbest_positions[i] = new_position
                    if new_fitness < self.global_best_fitness:
                        self.global_best_fitness = new_fitness
                        self.global_best_position = new_position

                self.population[i] = new_position

        return self.global_best_position