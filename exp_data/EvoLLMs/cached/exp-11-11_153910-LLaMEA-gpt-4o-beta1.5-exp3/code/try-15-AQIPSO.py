import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive parameter
        self.c2 = 1.5 # Social parameter

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_inspired_update(self, particle, global_best):
        delta = np.random.uniform(size=self.dim)
        return (particle + global_best) / 2 + delta * (particle - global_best)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.population[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.population[i]))
                
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
                
                if np.random.rand() < 0.1:  # Introduce quantum-inspired update with a small probability
                    self.population[i] = self.quantum_inspired_update(self.population[i], self.global_best_position)
                
                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = self.population[i]

        return self.global_best_position