import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.best_personal_position = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.evaluations = 0
        self.omega = 0.5
        self.phi_p = 1.5
        self.phi_g = 1.5

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_particle(self, i):
        r_p = np.random.rand(self.dim)
        r_g = np.random.rand(self.dim)
        cognitive_component = self.phi_p * r_p * (self.best_personal_position[i] - self.population[i])
        social_component = self.phi_g * r_g * (self.best_global_position - self.population[i])
        self.velocities[i] = self.omega * self.velocities[i] + cognitive_component + social_component
        self.population[i] += self.velocities[i]
        self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

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
            for i in range(self.pop_size):
                self.update_particle(i)
                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = current_fitness
                    self.best_personal_position[i] = self.population[i]
                if current_fitness < self.best_global_fitness:
                    self.best_global_fitness = current_fitness
                    self.best_global_position = self.population[i]

        return self.best_global_position