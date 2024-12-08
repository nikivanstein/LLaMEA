import numpy as np

class ACPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, self.dim))
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = np.full(self.pop_size, float('inf'))
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.pbest_fitness[i] = fitness
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.population[i]
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # Update velocity
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.population[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.population[i]))
                
                # Update position
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                fitness = self.evaluate(func, self.population[i])
                
                # Update personal best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.population[i]
                
                # Update global best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.population[i]

        return self.gbest_position