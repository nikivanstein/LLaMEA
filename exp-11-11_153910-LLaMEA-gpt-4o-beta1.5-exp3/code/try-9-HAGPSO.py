import numpy as np

class HAGPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.4 # Personal attraction coefficient
        self.c2 = 1.4 # Global attraction coefficient

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
                self.personal_best_position[i] = self.population[i]
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocity[i] = (self.w * self.velocity[i] +
                                    self.c1 * r1 * (self.personal_best_position[i] - self.population[i]) +
                                    self.c2 * r2 * (self.global_best_position - self.population[i]))
                # Update position
                self.population[i] += self.velocity[i]
                # Crossover and Mutation
                if np.random.rand() < 0.5:
                    partner_idx = np.random.choice([idx for idx in range(self.pop_size) if idx != i])
                    crossover_point = np.random.randint(1, self.dim)
                    self.population[i][:crossover_point] = self.population[partner_idx][:crossover_point]
                if np.random.rand() < 0.1:
                    mutation_idx = np.random.randint(self.dim)
                    self.population[i][mutation_idx] += np.random.normal(0, 1)

                # Ensure bounds
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate
                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_position[i] = self.population[i]
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position