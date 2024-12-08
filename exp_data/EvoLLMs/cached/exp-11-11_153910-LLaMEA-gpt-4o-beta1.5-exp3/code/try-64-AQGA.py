import numpy as np

class AQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_superposition(self, individual):
        theta = np.pi * np.random.rand(self.dim)
        return np.tan(theta) * individual

    def adaptive_crossover(self, parent1, parent2):
        beta = np.random.rand()
        return beta * parent1 + (1 - beta) * parent2

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            new_population = []
            for i in range(self.pop_size // 2):
                parent1, parent2 = self.population[np.random.choice(self.pop_size, 2, replace=False)]
                
                offspring1 = self.adaptive_crossover(parent1, parent2)
                offspring2 = self.adaptive_crossover(parent2, parent1)
                
                offspring1 = self.quantum_superposition(offspring1)
                offspring2 = self.quantum_superposition(offspring2)
                
                offspring1 = np.clip(offspring1, self.lower_bound, self.upper_bound)
                offspring2 = np.clip(offspring2, self.lower_bound, self.upper_bound)
                
                new_population.extend([offspring1, offspring2])

            for i in range(self.pop_size):
                self.population[i] = new_population[i]
                self.fitness[i] = self.evaluate(func, self.population[i])
                if self.fitness[i] < self.best_global_fitness:
                    self.best_global_fitness = self.fitness[i]
                    self.best_global_position = self.population[i]

        return self.best_global_position