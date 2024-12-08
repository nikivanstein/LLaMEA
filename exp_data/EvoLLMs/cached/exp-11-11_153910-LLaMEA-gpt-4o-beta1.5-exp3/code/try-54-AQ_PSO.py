import numpy as np

class AQ_PSO:
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
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_velocity_position(self, i):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
        social_component = self.c2 * r2 * (self.best_global_position - self.population[i])
        quantum_component = np.random.normal(0, 1, self.dim) * (self.best_global_position - self.population[i])
        
        self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + quantum_component
        self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = self.population[i]
            if fitness < self.best_global_fitness:
                self.best_global_fitness = fitness
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.update_velocity_position(i)
                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]
                    if fitness < self.best_global_fitness:
                        self.best_global_fitness = fitness
                        self.best_global_position = self.population[i]

        return self.best_global_position